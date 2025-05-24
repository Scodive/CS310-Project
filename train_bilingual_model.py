import glob
import os
import json
from text_classifier import TextClassificationTrainer # 确保 text_classifier.py 在PYTHONPATH中或者同目录下

def train_bilingual_model():
    """训练并评估一个中英双语文本分类模型"""

    print("开始训练中英双语文本分类模型...")

    # 1. 定义数据路径
    # 中文数据路径
    chinese_ai_data_pattern = "/home/hljiang/CS310-Natural_Language_Processing/generated/zh_qwen2/*.json"
    chinese_human_data_pattern = "/home/hljiang/CS310-Natural_Language_Processing/human/zh_unicode/*.json"

    # 英文数据路径
    english_base_data_dir = "/home/hljiang/CS310-Natural_Language_Processing/ghostbuster-data/essay"
    english_human_data_pattern = os.path.join(english_base_data_dir, "human", "*.txt")
    
    # 查找所有英文AI子目录
    english_ai_source_folders = [os.path.join(english_base_data_dir, d) for d in os.listdir(english_base_data_dir) 
                                 if os.path.isdir(os.path.join(english_base_data_dir, d)) and d != "human"]
    
    # 2. 收集所有数据文件
    # 中文文件
    chinese_ai_files = glob.glob(chinese_ai_data_pattern)
    chinese_human_files = glob.glob(chinese_human_data_pattern)

    # 英文文件
    english_ai_files = []
    for folder in english_ai_source_folders:
        english_ai_files.extend(glob.glob(os.path.join(folder, "*.txt")))
    english_human_files = glob.glob(english_human_data_pattern)

    # 合并AI和人类文件列表
    all_ai_files = chinese_ai_files + english_ai_files
    all_human_files = chinese_human_files + english_human_files
    
    model_name = 'bert-base-multilingual-cased' # 双语模型
    output_model_dir = './saved_models/bilingual_model' # 模型保存路径

    # 3. 检查数据文件
    if not all_ai_files:
        print(f"警告: 未找到任何AI生成的数据文件 (中英文合计)。")
    if not all_human_files:
        print(f"警告: 未找到任何人类书写的数据文件 (中英文合计)。")

    if not all_ai_files and not all_human_files:
        print("错误: AI和人类数据文件均未找到。无法继续训练。")
        return

    print(f"找到 {len(all_ai_files)} 个AI生成的总数据文件 ({len(chinese_ai_files)} 中文JSON, {len(english_ai_files)} 英文TXT)。")
    print(f"找到 {len(all_human_files)} 个人类书写的总数据文件 ({len(chinese_human_files)} 中文JSON, {len(english_human_files)} 英文TXT)。")

    # 4. 初始化训练器
    print(f"初始化模型: {model_name}")
    trainer = TextClassificationTrainer(model_name=model_name, max_length=512)

    # 5. 加载数据
    try:
        print("加载数据...")
        # load_data_from_specific_sources 能自动处理混合文件类型
        texts, labels = trainer.load_data_from_specific_sources(
            ai_generated_files=all_ai_files,
            human_written_files=all_human_files
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

    # 6. 准备数据 (分割训练集和测试集)
    print("准备数据 (分割训练集/测试集)...")
    try:
        train_dataset, test_dataset = trainer.prepare_data(texts, labels, test_size=0.2, random_state=42)
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        if not train_dataset or not test_dataset:
            print("错误：未能成功创建训练集或测试集。")
            return
    except Exception as e:
        print(f"准备数据时出错: {e}")
        return

    # 7. 训练模型
    print("开始训练模型...")
    val_dataset_for_train = test_dataset if len(test_dataset) > 0 else None
    if val_dataset_for_train is None and len(test_dataset) == 0:
        print("警告: 测试集为空，训练期间将不进行验证。")

    try:
        training_history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset_for_train,
            epochs=3,       # 可根据需要调整
            batch_size=256,  # 多语言数据量可能较大，batch_size可能需要调整
            learning_rate=2e-5
        )
        print("模型训练完成。")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        if "CUDA out of memory" in str(e):
            print("检测到CUDA内存不足错误。尝试减小 batch_size 或 max_length。")
        return

    # 8. 评估模型
    if len(test_dataset) > 0:
        print("开始评估模型...")
        try:
            from torch.utils.data import DataLoader
            final_test_loader = DataLoader(test_dataset, batch_size=16)
            eval_loss, accuracy, detailed_metrics = trainer.evaluate(final_test_loader)
            print("模型评估完成。")
            print(f"  测试集损失: {eval_loss:.4f}")
            print(f"  测试集准确率: {accuracy:.4f}")
            print(f"  测试集F1分数: {detailed_metrics['f1']:.4f}")
            print(f"  测试集AUC-ROC: {detailed_metrics['auc_roc']:.4f}")

            eval_results_path = os.path.join(output_model_dir, 'evaluation_results.json')
            os.makedirs(output_model_dir, exist_ok=True)
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

    # 9. 保存模型
    print(f"保存模型到: {output_model_dir}")
    try:
        trainer.save_model(output_model_dir)
        print("模型保存成功。")
    except Exception as e:
        print(f"保存模型时发生错误: {e}")

    print("中英双语模型训练流程结束。")

if __name__ == "__main__":
    train_bilingual_model() 