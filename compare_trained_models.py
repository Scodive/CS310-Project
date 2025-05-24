import glob
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torch.utils.data import DataLoader
from text_classifier import TextClassificationTrainer, TextDataset # 确保导入

# 设置Matplotlib支持中文显示
# 您可能需要确保系统中有名为 'SimHei' 的字体，或者替换为其他可用的中文字体
try:
    font_path = fm.findfont(fm.FontProperties(family='SimHei'))
    if font_path:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:
        print("警告: 未找到'SimHei'字体，中文可能无法正确显示。尝试下载并安装'SimHei'字体，或指定其他可用中文字体路径。")
        # 尝试寻找其他常见中文字体作为备选
        common_chinese_fonts = ['Microsoft YaHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
        for font_name in common_chinese_fonts:
            try:
                font_path_alt = fm.findfont(fm.FontProperties(family=font_name))
                if font_path_alt:
                    plt.rcParams['font.sans-serif'] = [font_name]
                    print(f"已切换到备选字体: {font_name}")
                    break
            except Exception:
                continue
except Exception as e:
    print(f"设置中文字体时出错: {e}")
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def get_model_parameter_count(model):
    """计算模型的参数数量"""
    # 如果模型是DataParallel包装的，要从.module访问实际模型
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    return sum(p.numel() for p in actual_model.parameters() if p.requires_grad)

def prepare_test_data_from_files(trainer_for_tokenizer, ai_files, human_files, lang_description):
    """从文件加载并准备测试数据"""
    print(f"准备 {lang_description} 测试数据...")
    try:
        texts, labels = trainer_for_tokenizer.load_data_from_specific_sources(
            ai_generated_files=ai_files,
            human_written_files=human_files
        )
        if not texts:
            print(f"警告: 未能为 {lang_description} 测试数据加载任何文本。")
            return None, [], 0, 0
        
        # 使用传入的trainer的tokenizer创建TextDataset
        dataset = TextDataset(texts, labels, trainer_for_tokenizer.tokenizer, trainer_for_tokenizer.max_length)
        print(f"为 {lang_description} 测试数据加载了 {len(texts)} 条样本。 AI: {labels.count(1)}, Human: {labels.count(0)}.")
        return dataset, labels, labels.count(1), labels.count(0)
    except Exception as e:
        print(f"为 {lang_description} 测试数据准备数据时出错: {e}")
        return None, [], 0, 0

def evaluate_model_on_dataset(trainer, test_dataset, batch_size=16):
    """在给定数据集上评估模型并测量资源"""
    if not test_dataset or len(test_dataset) == 0:
        return {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc_roc': 0,
            'inference_time_seconds': 0, 'peak_gpu_memory_mb': 0, 'error': 'No test data'
        }

    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 推理时间
    start_time = time.time()
    # 确保在评估前模型在正确的设备上
    trainer.model.to(trainer.device)
    # 评估指标
    # evaluate方法现在返回 avg_loss, accuracy, detailed_metrics_dict
    _, _, metrics = trainer.evaluate(dataloader, verbose=False) 
    end_time = time.time()
    inference_time = end_time - start_time

    # GPU内存 (如果可用)
    peak_gpu_memory_mb = 0
    if torch.cuda.is_available() and trainer.device.type == 'cuda':
        # DataParallel可能将模型分布在多个GPU，这里我们获取第一个GPU（通常是device 0）的统计信息
        # 或者，如果trainer.device被特定设置，则使用该device
        device_for_mem = trainer.device if trainer.device.index is not None else torch.device('cuda', 0)
        torch.cuda.reset_peak_memory_stats(device_for_mem)
        # 再次运行一小部分预测以捕获内存，因为evaluate可能已经清除了某些状态
        # 或者依赖evaluate内部的内存峰值（如果它做了完整的推理）
        # 为了简化，我们这里仅报告evaluate调用后的峰值，这可能不完全是推理峰值，但具有代表性
        # 更准确的做法是在evaluate内部或predict内部进行精确测量
        peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device_for_mem)
        peak_gpu_memory_mb = peak_gpu_memory_bytes / (1024 * 1024)
        print(f"模型 {trainer.model_name} 在数据集上评估后的峰值GPU内存 ({device_for_mem}): {peak_gpu_memory_mb:.2f} MB")

    return {
        'accuracy': metrics.get('accuracy', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1': metrics.get('f1', 0),
        'auc_roc': metrics.get('auc_roc', 0),
        'inference_time_seconds': inference_time,
        'peak_gpu_memory_mb': peak_gpu_memory_mb
    }

def main_comparison():
    """执行模型对比"""
    model_configs = [
        {'name': 'bert-base-chinese', 'path': './saved_models/chinese_model', 'type': 'Chinese'},
        {'name': 'bert-base-uncased', 'path': './saved_models/english_model', 'type': 'English'},
        {'name': 'bert-base-multilingual-cased', 'path': './saved_models/bilingual_model', 'type': 'Bilingual'}
    ]

    comparison_results = {"model_comparison": {}, "test_data_summary": {}}
    
    # 准备一个通用的trainer实例，仅用于加载测试数据时的tokenizer
    # (实际评估时会为每个模型加载自己的trainer)
    # 使用多语言tokenizer作为通用tokenizer，因为它最通用
    try:
        generic_tokenizer_trainer = TextClassificationTrainer(model_name='bert-base-multilingual-cased')
    except Exception as e:
        print(f"错误：无法初始化通用tokenizer的TextClassificationTrainer: {e}")
        print("请确保Hugging Face模型 'bert-base-multilingual-cased' 可以被下载或已缓存。")
        return

    # 定义测试数据文件 (少量，用于快速对比)
    # 中文测试数据
    test_zh_ai_files = glob.glob("/home/hljiang/CS310-Natural_Language_Processing/generated/zh_qwen2/news-zh.qwen2-72b-base.json")[:1]
    test_zh_human_files = glob.glob("/home/hljiang/CS310-Natural_Language_Processing/human/zh_unicode/news-zh.json")[:1]
    
    # 英文测试数据
    eng_base_dir = "/home/hljiang/CS310-Natural_Language_Processing/ghostbuster-data/essay"
    test_en_human_files = glob.glob(os.path.join(eng_base_dir, "human", "*.txt"))[:50] # 取50个
    eng_ai_folders = [os.path.join(eng_base_dir, d) for d in os.listdir(eng_base_dir) 
                      if os.path.isdir(os.path.join(eng_base_dir, d)) and d != "human"]
    test_en_ai_files = []
    for folder in eng_ai_folders:
        test_en_ai_files.extend(glob.glob(os.path.join(folder, "*.txt")))
    test_en_ai_files = test_en_ai_files[:50] # 取50个

    # 准备各测试集
    # 使用 generic_tokenizer_trainer 来获取tokenizer进行数据准备
    # (这里假设所有模型的分词策略对于测试数据的格式化是可以接受的，或者TextDataset的实现是健壮的)
    # 实际上，每个模型应该用自己的tokenizer来处理送入它的数据，但TextDataset在创建时接收tokenizer
    # 为了在evaluate_model_on_dataset中保持一致，我们为每个模型加载自己的trainer和tokenizer

    datasets_to_evaluate = {}
    # 为了能用各自模型的tokenizer，我们在加载模型后，再用该模型的tokenizer创建TextDataset
    # 这里先定义文件路径，后面在模型循环中创建具体的TextDataset实例
    test_data_definitions = {
        "chinese_data": {
            "ai_files": test_zh_ai_files, 
            "human_files": test_zh_human_files, 
            "description": "全中文"
        },
        "english_data": {
            "ai_files": test_en_ai_files, 
            "human_files": test_en_human_files, 
            "description": "全英文"
        },
        "mixed_data": {
            "ai_files": test_zh_ai_files + test_en_ai_files, 
            "human_files": test_zh_human_files + test_en_human_files, 
            "description": "中英混合"
        }
    }

    for model_config in model_configs:
        model_name = model_config['name']
        model_path = model_config['path']
        print(f"\n--- 正在处理模型: {model_name} --- PATH: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"警告: 模型路径 {model_path} 不存在，跳过此模型。")
            comparison_results["model_comparison"][model_name] = {
                "parameter_count": "N/A",
                "error": "Model path not found",
                "test_results": {}
            }
            continue

        trainer = TextClassificationTrainer(model_name=model_name) # 使用模型自身的name初始化
        try:
            trainer.load_model(model_path)
            print(f"成功加载模型: {model_name}")
        except Exception as e:
            print(f"加载模型 {model_name} 从 {model_path} 失败: {e}")
            comparison_results["model_comparison"][model_name] = {
                "parameter_count": "N/A",
                "error": f"Failed to load model: {str(e)}",
                "test_results": {}
            }
            continue

        param_count = get_model_parameter_count(trainer.model)
        comparison_results["model_comparison"][model_name] = {
            "parameter_count": param_count,
            "model_type": model_config['type'],
            "test_results": {}
        }
        print(f"模型参数量: {param_count}")

        for ds_name, ds_def in test_data_definitions.items():
            print(f"  评估 {model_name} 在 {ds_def['description']} 数据集上...")
            
            # 使用当前模型的trainer (包含其特定tokenizer) 来准备TextDataset
            current_test_dataset, current_labels, ai_count, human_count = prepare_test_data_from_files(
                trainer, ds_def['ai_files'], ds_def['human_files'], ds_def['description']
            )
            
            if ds_name not in comparison_results["test_data_summary"] and current_test_dataset:
                 comparison_results["test_data_summary"][ds_name] = {
                     "num_samples": len(current_test_dataset),
                     "ai_samples": ai_count,
                     "human_samples": human_count
                 }
            elif not current_test_dataset and ds_name not in comparison_results["test_data_summary"]:
                 comparison_results["test_data_summary"][ds_name] = {
                     "num_samples": 0, "ai_samples": 0, "human_samples": 0, "error": "Failed to load"
                 }

            eval_results = evaluate_model_on_dataset(trainer, current_test_dataset, batch_size=32) # 使用统一的batch_size进行评估
            comparison_results["model_comparison"][model_name]["test_results"][ds_name] = eval_results
            print(f"    结果 ({ds_def['description']}): Acc: {eval_results['accuracy']:.4f}, F1: {eval_results['f1']:.4f}, Time: {eval_results['inference_time_seconds']:.2f}s, GPU Mem: {eval_results['peak_gpu_memory_mb']:.2f}MB")

    # 保存结果到JSON
    report_path = 'trained_models_comparison_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=4)
    print(f"\n比较报告已保存到: {report_path}")

    # 生成图表
    plot_comparison_results(comparison_results)

def plot_comparison_results(results):
    """生成对比图表"""
    if not results or "model_comparison" not in results:
        print("没有足够的对比结果来生成图表。")
        return

    model_names = list(results["model_comparison"].keys())
    # 过滤掉加载失败的模型
    valid_models_data = {name: data for name, data in results["model_comparison"].items() if "error" not in data}
    if not valid_models_data:
        print("所有模型都加载失败或没有有效数据，无法生成图表。")
        return
    
    model_names = list(valid_models_data.keys())
    num_models = len(model_names)

    dataset_names = list(results.get("test_data_summary", {}).keys())
    if not dataset_names:
         # 尝试从第一个有效模型的结果中获取数据集名称
        first_model_key = next(iter(valid_models_data))
        if valid_models_data[first_model_key]["test_results"]:
            dataset_names = list(valid_models_data[first_model_key]["test_results"].keys())
        else:
            print("无法确定数据集名称用于绘图。")
            return
            
    num_datasets = len(dataset_names)

    # 准备绘图数据
    accuracies = np.zeros((num_models, num_datasets))
    f1_scores = np.zeros((num_models, num_datasets))
    inference_times = np.zeros((num_models, num_datasets))
    param_counts = []
    gpu_memories = np.zeros((num_models, num_datasets))

    dataset_display_names = [results["test_data_summary"].get(ds, {}).get("description", ds) for ds in dataset_names]

    for i, model_name in enumerate(model_names):
        model_data = valid_models_data[model_name]
        param_counts.append(model_data.get("parameter_count", 0) / 1e6) # 转为百万单位
        for j, ds_name in enumerate(dataset_names):
            res = model_data["test_results"].get(ds_name, {})
            accuracies[i, j] = res.get('accuracy', 0)
            f1_scores[i, j] = res.get('f1', 0)
            inference_times[i, j] = res.get('inference_time_seconds', 0)
            gpu_memories[i, j] = res.get('peak_gpu_memory_mb', 0)
    
    num_metrics_datasets = 3 # Acc, F1, Time
    if torch.cuda.is_available():
        num_metrics_total = num_metrics_datasets + 2 # + Params, + GPU Mem
        fig_height = 18
    else:
        num_metrics_total = num_metrics_datasets + 1 # + Params
        fig_height = 15

    fig, axs = plt.subplots(num_metrics_total, 1, figsize=(12, fig_height))
    fig.tight_layout(pad=4.0)
    
    x = np.arange(num_datasets)
    width = 0.25

    plot_idx = 0

    # 准确率
    for i, model_name in enumerate(model_names):
        axs[plot_idx].bar(x + i * width - (num_models-1)*width/2, accuracies[i, :], width, label=model_name)
    axs[plot_idx].set_ylabel('准确率 (Accuracy)')
    axs[plot_idx].set_title('各模型在不同测试集上的准确率')
    axs[plot_idx].set_xticks(x)
    axs[plot_idx].set_xticklabels(dataset_display_names)
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, axis='y')
    plot_idx += 1

    # F1分数
    for i, model_name in enumerate(model_names):
        axs[plot_idx].bar(x + i * width - (num_models-1)*width/2, f1_scores[i, :], width, label=model_name)
    axs[plot_idx].set_ylabel('F1分数 (F1-Score)')
    axs[plot_idx].set_title('各模型在不同测试集上的F1分数')
    axs[plot_idx].set_xticks(x)
    axs[plot_idx].set_xticklabels(dataset_display_names)
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, axis='y')
    plot_idx += 1

    # 推理时间
    for i, model_name in enumerate(model_names):
        axs[plot_idx].bar(x + i * width - (num_models-1)*width/2, inference_times[i, :], width, label=model_name)
    axs[plot_idx].set_ylabel('推理时间 (秒)')
    axs[plot_idx].set_title('各模型在不同测试集上的推理时间')
    axs[plot_idx].set_xticks(x)
    axs[plot_idx].set_xticklabels(dataset_display_names)
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, axis='y')
    plot_idx += 1

    # 参数量
    model_ticks = np.arange(num_models)
    axs[plot_idx].bar(model_ticks, param_counts, width * num_models * 0.8, color=['skyblue', 'lightcoral', 'lightgreen'][:num_models])
    axs[plot_idx].set_ylabel('参数量 (百万)')
    axs[plot_idx].set_title('各模型的参数量')
    axs[plot_idx].set_xticks(model_ticks)
    axs[plot_idx].set_xticklabels(model_names, rotation=15, ha='right')
    axs[plot_idx].grid(True, axis='y')
    plot_idx += 1

    # GPU内存 (如果可用)
    if torch.cuda.is_available():
        for i, model_name in enumerate(model_names):
            axs[plot_idx].bar(x + i * width - (num_models-1)*width/2, gpu_memories[i, :], width, label=model_name)
        axs[plot_idx].set_ylabel('峰值GPU内存 (MB)')
        axs[plot_idx].set_title('各模型在不同测试集上的峰值GPU内存')
        axs[plot_idx].set_xticks(x)
        axs[plot_idx].set_xticklabels(dataset_display_names)
        axs[plot_idx].legend()
        axs[plot_idx].grid(True, axis='y')
        plot_idx += 1

    plt.savefig('trained_models_comparison_plots.png', dpi=300, bbox_inches='tight')
    print("\n对比图表已保存到: trained_models_comparison_plots.png")
    # plt.show() # 在脚本中通常不显示，除非需要交互

if __name__ == "__main__":
    main_comparison() 