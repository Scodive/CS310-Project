import glob
import os
import json
from text_classifier import TextClassificationTrainer # 确保 text_classifier.py 在PYTHONPATH中或者同目录下

def train_english_model():
    """训练并评估英文文本分类模型"""

    print("开始训练英文文本分类模型...")

    # 1. 定义数据路径和模型名称
    base_data_dir = "/home/hljiang/CS310-Natural_Language_Processing/ghostbuster-data/essay"
    human_data_pattern = os.path.join(base_data_dir, "human", "*.txt")
    
    # 查找所有AI子目录 (除human之外的所有子目录)
    ai_source_folders = [os.path.join(base_data_dir, d) for d in os.listdir(base_data_dir) 
                         if os.path.isdir(os.path.join(base_data_dir, d)) and d != "human"]
    
    ai_files = []
    for folder in ai_source_folders:
        ai_files.extend(glob.glob(os.path.join(folder, "*.txt")))

    human_files = glob.glob(human_data_pattern)
    
    model_name = 'bert-base-uncased' # 英文模型
    output_model_dir = './saved_models/english_model' # 模型保存路径

    # 2. 检查数据文件
    if not ai_files:
        print(f"警告: 在 {base_data_dir} 下的AI子目录中未找到任何 .txt 文件。")
    if not human_files:
        print(f"警告: 在 {human_data_pattern} 未找到人类书写的 .txt 文件。")

    if not ai_files and not human_files:
        print("错误: AI和人类数据文件均未找到。无法继续训练。")
        return

    print(f"找到 {len(ai_files)} 个AI生成的 .txt 数据文件。")
    # print("AI文件示例:", ai_files[:5]) # 打印一些示例文件
    print(f"找到 {len(human_files)} 个人类书写的 .txt 数据文件。")
    # print("人类文件示例:", human_files[:5]) # 打印一些示例文件

    # 3. 初始化训练器
    print(f"初始化模型: {model_name}")
    trainer = TextClassificationTrainer(model_name=model_name, max_length=512)

    # 4. 加载数据
    try:
        print("加载数据...")
        texts, labels = trainer.load_data_from_specific_sources(
            ai_generated_files=ai_files,
            human_written_files=human_files
        )
        print(f"成功加载 {len(texts)} 条文本数据 ({labels.count(1)} AI, {labels.count(0)} Human)。")
        if not texts:
            print("错误：加载数据后未获取到任何文本。请检查数据文件内容和加载逻辑。")
            return
    except ValueError as e:
        print(f"加载数据时出错: {e}")
        return
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}")
        return

    # 5. 准备数据 (分割训练集和测试集)
    print("准备数据 (分割训练集/测试集)...")
    try:
        train_dataset, test_dataset = trainer.prepare_data(texts, labels, test_size=0.2, random_state=42)
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        if not train_dataset or not test_dataset:
            print("错误：未能成功创建训练集或测试集。")
            return
    except Exception as e: # 捕获更广泛的错误，因为prepare_data内部处理了ValueError
        print(f"准备数据时出错: {e}")
        return

    # 6. 训练模型
    print("开始训练模型...")
    val_dataset_for_train = test_dataset if len(test_dataset) > 0 else None
    if val_dataset_for_train is None and len(test_dataset) == 0:
        print("警告: 测试集为空，训练期间将不进行验证。")

    try:
        training_history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset_for_train,
            epochs=3,       # 可根据需要调整
            batch_size=8,   # 英文文本可能较长，适当减小batch_size
            learning_rate=2e-5
        )
        print("模型训练完成。")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        # 在发生GPU OOM等错误时，打印更多帮助信息
        if "CUDA out of memory" in str(e):
            print("检测到CUDA内存不足错误。尝试减小 batch_size 或 max_length。")
        return

    # 7. 评估模型
    if len(test_dataset) > 0:
        print("开始评估模型...")
        try:
            from torch.utils.data import DataLoader
            final_test_loader = DataLoader(test_dataset, batch_size=8)
            eval_loss, accuracy, detailed_metrics = trainer.evaluate(final_test_loader)
            print("模型评估完成。")
            print(f"  测试集损失: {eval_loss:.4f}")
            print(f"  测试集准确率: {accuracy:.4f}")
            print(f"  测试集F1分数: {detailed_metrics['f1']:.4f}")
            print(f"  测试集AUC-ROC: {detailed_metrics['auc_roc']:.4f}")

            eval_results_path = os.path.join(output_model_dir, 'evaluation_results.json')
            os.makedirs(output_model_dir, exist_ok=True)
            # detailed_metrics 包含 numpy array，需要转换为 list 才能 JSON 序列化
            for key, value in detailed_metrics.items():
                if isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float, __import__('numpy').integer, __import__('numpy').floating)):
                    detailed_metrics[key] = [float(v) for v in value]

            with open(eval_results_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_metrics, f, ensure_ascii=False, indent=4)
            print(f"详细评估结果已保存到: {eval_results_path}")

        except Exception as e:
            print(f"评估过程中发生错误: {e}")
    else:
        print("测试集为空，跳过评估。")

    # 8. 保存模型
    print(f"保存模型到: {output_model_dir}")
    try:
        trainer.save_model(output_model_dir)
        print("模型保存成功。")
    except Exception as e:
        print(f"保存模型时发生错误: {e}")

    print("英文模型训练流程结束。")

if __name__ == "__main__":
    train_english_model() 