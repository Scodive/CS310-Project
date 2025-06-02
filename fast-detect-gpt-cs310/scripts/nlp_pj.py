# Copyright (c) Guangsheng Bao.
# Modified by Assistant for batch processing and metrics
import os
import json
import time
import argparse
import glob
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                             auc)
import matplotlib.pyplot as plt

from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from scipy.stats import norm


def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob


class FastDetectGPT:
    def __init__(self, args):
        print("\n[初始化阶段] 开始构建 FastDetectGPT 检测器")
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        print(f"Step 1/2: 加载评分模型 {args.scoring_model_name}...")
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        print(f"评分模型加载完成，设备: {self.scoring_model.device}")
        if args.sampling_model_name != args.scoring_model_name:
            print(f"Step 2/2: 加载采样模型 {args.sampling_model_name}...")
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()
            print(f"采样模型加载完成，设备: {self.sampling_model.device}")
        print("[初始化阶段] FastDetectGPT 准备就绪\n")
        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'EleutherAI/gpt-neo-2.7B_EleutherAI/gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983,
                                                                'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        self.classifier = distrib_params[key]

    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True,
                                           return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True,
                                                    return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken


def evaluate_dataset(detector, texts, true_labels):
    """批量处理数据集并返回结果"""
    probs = []
    crits = []
    n_tokens_list = []
    total_time = 0
    valid_samples = 0

    for i, text in enumerate(texts):
        try:
            start_time = time.time()
            prob, crit, n_tokens = detector.compute_prob(text)
            total_time += time.time() - start_time

            probs.append(prob)
            crits.append(crit)
            n_tokens_list.append(n_tokens)
            valid_samples += 1

            # 每处理10个样本打印一次进度
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(texts)} 个样本")
        except Exception as e:
            print(f"处理样本 {i + 1} 失败: {str(e)}")
            probs.append(np.nan)
            crits.append(np.nan)
            n_tokens_list.append(0)

    # 过滤无效样本
    mask = ~np.isnan(probs)
    probs = np.array(probs)[mask]
    crits = np.array(crits)[mask]
    true_labels = np.array(true_labels)[mask]

    return {
        'probs': probs,
        'crits': crits,
        'true_labels': true_labels,
        'avg_time': total_time / valid_samples if valid_samples > 0 else 0,
        'total_samples': len(texts),
        'valid_samples': valid_samples
    }


def calculate_metrics(results):
    """计算评估指标"""
    probs = results['probs']
    true_labels = results['true_labels']

    if len(np.unique(true_labels)) < 2:
        return {'error': '需要两类样本来计算指标'}

    # 二分类预测
    pred_labels = (probs >= 0.5).astype(int)

    # 基础指标
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels),
        'roc_auc': roc_auc_score(true_labels, probs)
    }

    # ROC曲线数据
    fpr, tpr, _ = roc_curve(true_labels, probs)
    metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

    # PR曲线数据
    precision, recall, _ = precision_recall_curve(true_labels, probs)
    metrics['pr_curve'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'auc': auc(recall, precision)
    }

    return metrics


def plot_curves(metrics, output_dir='results'):
    """绘制ROC和PR曲线"""
    os.makedirs(output_dir, exist_ok=True)

    # ROC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['roc_curve']['fpr'], metrics['roc_curve']['tpr'],
             label=f"ROC曲线 (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # PR曲线
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['pr_curve']['recall'], metrics['pr_curve']['precision'],
             label=f"PR曲线 (AUC = {metrics['pr_curve']['auc']:.4f})")
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()


def _load_json_files_from_list(file_paths, text_key='text'):
    """从JSON文件路径列表中加载文本"""
    all_texts = []
    for file_path in file_paths:
        current_file_texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 优先处理 {"input": {"0": "text_a", "1": "text_b", ...}} 结构
                if isinstance(data, dict) and "input" in data and isinstance(data.get("input"), dict):
                    for _, text_value in data["input"].items():
                        if isinstance(text_value, str):
                            current_file_texts.append(text_value)
                # 处理JSON对象列表: [{text_key: "value"}, ...]
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and text_key in item and isinstance(item[text_key], str):
                            current_file_texts.append(item[text_key])
                        elif isinstance(item, str):  # 处理字符串列表: ["text1", "text2"]
                            current_file_texts.append(item)
                # 处理单个JSON对象，其中包含指定的text_key: {text_key: "value"}
                elif isinstance(data, dict) and text_key in data and isinstance(data[text_key], str):
                    current_file_texts.append(data[text_key])

            if current_file_texts:
                all_texts.extend(current_file_texts)
                print(f"成功加载: {file_path}, 提取到 {len(current_file_texts)} 条文本")
            else:
                if os.path.getsize(file_path) > 0:
                    print(f"警告: 未从 {file_path} 提取到任何文本")
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {file_path} - {e}")
        except Exception as e:
            print(f"加载文件错误: {file_path} - {e}")
    return all_texts


def _load_plain_text_files_from_list(file_paths):
    """从纯文本文件路径列表中加载文本"""
    all_texts = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    all_texts.append(content.strip())
            print(f"成功加载: {file_path}, 提取到1条文本")
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
        except Exception as e:
            print(f"加载文件错误: {file_path} - {e}")
    return all_texts


def load_data_from_specific_sources(ai_files, human_files, ai_text_key='output', human_text_key='output'):
    ai_texts = []
    human_texts = []

    # 处理AI生成的文件
    ai_json_files = [f for f in ai_files if f.lower().endswith('.json')]
    ai_txt_files = [f for f in ai_files if f.lower().endswith('.txt')]
    ai_other_files = [f for f in ai_files if not f.lower().endswith(('.json', '.txt'))]

    if ai_json_files:
        ai_texts.extend(_load_json_files_from_list(ai_json_files, text_key=ai_text_key))
    if ai_txt_files:
        ai_texts.extend(_load_plain_text_files_from_list(ai_txt_files))
    if ai_other_files:
        print(f"警告：跳过 {len(ai_other_files)} 个不支持的非JSON/TXT文件")

    # 处理人类写的文件
    human_json_files = [f for f in human_files if f.lower().endswith('.json')]
    human_txt_files = [f for f in human_files if f.lower().endswith('.txt')]
    human_other_files = [f for f in human_files if not f.lower().endswith(('.json', '.txt'))]

    if human_json_files:
        human_texts.extend(_load_json_files_from_list(human_json_files, text_key=human_text_key))
    if human_txt_files:
        human_texts.extend(_load_plain_text_files_from_list(human_txt_files))
    if human_other_files:
        print(f"警告：跳过 {len(human_other_files)} 个不支持的非JSON/TXT文件")

    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)
    return texts, labels


def run_batch_evaluation(detector, texts, labels, args):
    """执行批量评估"""
    # 处理数据
    print(f"开始处理 {len(texts)} 个样本...")
    results = evaluate_dataset(detector, texts, labels)

    # 计算指标
    metrics = calculate_metrics(results)

    # 保存结果
    output_dir = 'detectgpt_results'
    os.makedirs(output_dir, exist_ok=True)

    # 创建可序列化的结果字典
    serializable_results = {
        'probs': results['probs'].tolist(),  # 转换为Python列表
        'crits': results['crits'].tolist(),  # 转换为Python列表
        'true_labels': results['true_labels'].tolist(),  # 转换为Python列表
        'avg_time': results['avg_time'],
        'total_samples': results['total_samples'],
        'valid_samples': results['valid_samples']
    }

    # 创建可序列化的指标字典
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            # 转换NumPy标量值为Python原生类型
            serializable_metrics[key] = value.item()
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            serializable_metrics[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    serializable_metrics[key][subkey] = subvalue.tolist()
                else:
                    serializable_metrics[key][subkey] = subvalue
        else:
            serializable_metrics[key] = value

    # 保存原始结果
    with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'results': serializable_results,
            'metrics': serializable_metrics
        }, f, indent=2, ensure_ascii=False)

    # 保存指标报告
    report = {
        'dataset_stats': {
            'total_samples': results['total_samples'],
            'valid_samples': results['valid_samples'],
            'avg_processing_time': f"{results['avg_time']:.4f} sec/sample"
        },
        'metrics': serializable_metrics
    }

    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 绘制曲线
    if 'error' not in metrics:
        plot_curves(metrics, output_dir)

    print("评估完成，结果保存在目录:", output_dir)
    print("主要指标:")
    print(f"准确率: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"精确率: {metrics.get('precision', 'N/A'):.4f}")
    print(f"召回率: {metrics.get('recall', 'N/A'):.4f}")
    print(f"F1分数: {metrics.get('f1', 'N/A'):.4f}")
    print(f"ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")


def collect_files(path):
    """收集文件路径，支持目录和单个文件"""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        return glob.glob(os.path.join(path, '**', '*.*'), recursive=True)
    else:
        print(f"警告: 路径 {path} 既不是文件也不是目录")
        return []


def run(args):
    # 1. 先初始化模型
    print("=== 第一步: 初始化模型 ===")
    detector = FastDetectGPT(args)

    # 2. 再加载数据
    print("\n=== 第二步: 加载数据 ===")
    print("开始收集文件路径")

    # 使用命令行参数指定的路径
    ai_dir = args.ai_dir
    human_dir = args.human_dir

    ai_files = collect_files(ai_dir)
    human_files = collect_files(human_dir)

    # 检查是否找到文件
    if not ai_files:
        print(f"警告: 在AI路径 {ai_dir} 下未找到任何文件")
    if not human_files:
        print(f"警告: 在人类路径 {human_dir} 下未找到任何文件")

    # 加载数据
    print("开始加载数据")
    texts, labels = load_data_from_specific_sources(ai_files, human_files)
    print(f"加载完成: {len(texts)} 条文本 ({len(ai_files)} AI文件, {len(human_files)} 人类文件)")

    # 3. 最后运行评估
    print("\n=== 第三步: 运行评估 ===")
    run_batch_evaluation(detector, texts, labels, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast-DetectGPT 批量检测工具')
    parser.add_argument('--sampling_model_name', type=str, default="EleutherAI/gpt-neo-2.7B",
                        help='采样模型名称')
    parser.add_argument('--scoring_model_name', type=str, default="EleutherAI/gpt-neo-2.7B",
                        help='评分模型名称')
    parser.add_argument('--device', type=str, default="cuda",
                        help='计算设备 (cuda 或 cpu)')
    parser.add_argument('--cache_dir', type=str, default="../cache",
                        help='模型缓存目录')
    parser.add_argument('--ai_dir', type=str,
                        default='../../face2_zh_json/generated/zh_qwen2/test.json',
                        help='AI生成文本的目录或文件路径')
    parser.add_argument('--human_dir', type=str,
                        default='../../face2_zh_json/human/zh_unicode/test.json',
                        help='人类撰写文本的目录或文件路径')

    args = parser.parse_args()

    # 打印配置信息
    print("=" * 50)
    print("Fast-DetectGPT 批量检测配置")
    print("=" * 50)
    print(f"采样模型: {args.sampling_model_name}")
    print(f"评分模型: {args.scoring_model_name}")
    print(f"计算设备: {args.device}")
    print(f"AI文本路径: {args.ai_dir}")
    print(f"人类文本路径: {args.human_dir}")
    print("=" * 50)

    run(args)