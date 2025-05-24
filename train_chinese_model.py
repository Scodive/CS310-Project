import glob
import os
import json
from text_classifier import TextClassificationTrainer # 确保 text_classifier.py 在PYTHONPATH中或者同目录下

def train_chinese_model():
    """训练并评估中文文本分类模型"""

    print("开始训练中文文本分类模型...")

    # 1. 定义数据路径和模型名称
    ai_data_pattern = "/home/hljiang/CS310-Natural_Language_Processing/generated/zh_qwen2/*.json"
    human_data_pattern = "/home/hljiang/CS310-Natural_Language_Processing/human/zh_unicode/*.json"
    model_name = 'bert-base-chinese'
    output_model_dir = './saved_models/chinese_model' # 模型保存路径

    # 2. 查找数据文件
    ai_files = glob.glob(ai_data_pattern)
    human_files = glob.glob(human_data_pattern)

    if not ai_files:
        print(f"警告: 在 {ai_data_pattern} 未找到AI生成的数据文件。")
        # 可以选择在这里引发错误或继续（如果允许部分数据缺失）
    if not human_files:
        print(f"警告: 在 {human_data_pattern} 未找到人类书写的数据文件。")
        # 可以选择在这里引发错误或继续

    if not ai_files and not human_files:
        print("错误: AI和人类数据文件均未找到。无法继续训练。")
        return

    print(f"找到 {len(ai_files)} 个AI生成的数据文件。")
    print(f"找到 {len(human_files)} 个人类书写的数据文件。")
    # 打印前几个文件名以供检查 (可选)
    # print("AI文件示例:", ai_files[:3])
    # print("人类文件示例:", human_files[:3])


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
        # 确保标签分布足够进行分层抽样
        if len(set(labels)) < 2 and len(texts) > 1 : # 如果只有一类标签，stratify会报错
            print("警告: 数据中只存在一种标签，无法进行分层抽样。将使用普通抽样。")
            train_dataset, test_dataset = trainer.prepare_data(texts, labels, test_size=0.2, random_state=42)
        elif not labels: # 没有标签数据
             print("错误：没有标签数据可用于训练。")
             return
        else:
            train_dataset, test_dataset = trainer.prepare_data(texts, labels, test_size=0.2, random_state=42)

        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        if not train_dataset or not test_dataset:
            print("错误：未能成功创建训练集或测试集。")
            return
    except ValueError as e:
        print(f"准备数据时出错（可能由于样本太少无法分层）: {e}")
        # 如果是因为样本太少无法分层，尝试不使用分层
        try:
            print("尝试不使用分层抽样...")
            train_dataset, test_dataset = trainer.prepare_data(texts, labels, test_size=0.2, random_state=42)
            print(f"训练集大小: {len(train_dataset)}")
            print(f"测试集大小: {len(test_dataset)}")
            if not train_dataset or not test_dataset:
                 print("错误：即使不使用分层抽样，也未能成功创建训练集或测试集。")
                 return
        except Exception as e_nostratify:
            print(f"不使用分层抽样也失败了: {e_nostratify}")
            return


    # 6. 训练模型
    print("开始训练模型...")
    # 确保测试集 (现在用作验证集) 不为空
    val_dataset_for_train = test_dataset if len(test_dataset) > 0 else None
    if val_dataset_for_train is None and len(test_dataset) == 0:
        print("警告: 测试集为空，训练期间将不进行验证。")

    try:
        training_history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset_for_train, # 使用测试集作为验证集
            epochs=3,       # 可根据需要调整
            batch_size=16,  # 可根据显存和数据量调整
            learning_rate=2e-5
        )
        print("模型训练完成。")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        return

    # 7. 评估模型
    if len(test_dataset) > 0:
        print("开始评估模型...")
        try:
            # 为评估创建一个新的DataLoader，以防之前的状态问题
            from torch.utils.data import DataLoader
            final_test_loader = DataLoader(test_dataset, batch_size=16)
            eval_loss, accuracy, detailed_metrics = trainer.evaluate(final_test_loader)
            print("模型评估完成。")
            print(f"  测试集损失: {eval_loss:.4f}")
            print(f"  测试集准确率: {accuracy:.4f}")
            print(f"  测试集F1分数: {detailed_metrics['f1']:.4f}")
            print(f"  测试集AUC-ROC: {detailed_metrics['auc_roc']:.4f}")

            # 保存详细评估结果
            eval_results_path = os.path.join(output_model_dir, 'evaluation_results.json')
            os.makedirs(output_model_dir, exist_ok=True)
            # detailed_metrics 包含 numpy array，需要转换为 list 才能 JSON 序列化
            for key, value in detailed_metrics.items():
                if isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float, np.integer, np.floating)):
                     # For lists/tuples of numbers like predictions, labels, probabilities
                    detailed_metrics[key] = [float(v) for v in value] # Convert numpy types to native Python types

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

    print("中文模型训练流程结束。")

if __name__ == "__main__":
    # 确保 text_classifier.py 可以在此脚本的目录中找到
    # 或者 CS310-Natural_Language_Processing 目录在 PYTHONPATH 中
    # 例如，如果此脚本在 CS310-Natural_Language_Processing 内部，可以直接导入
    # 如果 text_classifier.py 在上一级目录，可能需要调整 sys.path
    # import sys
    # sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 添加当前脚本所在目录到sys.path
    
    train_chinese_model() 