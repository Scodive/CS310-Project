# run_detection.py
import os
import json
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                             auc)
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set Chinese font support
mpl.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
mpl.rcParams['axes.unicode_minus'] = False  # Display minus sign correctly

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
        print("\n[Initialization] Building FastDetectGPT detector")
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        print(f"Step 1/2: Loading scoring model {args.scoring_model_name}...")
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        print(f"Scoring model loaded, device: {self.scoring_model.device}")
        if args.sampling_model_name != args.scoring_model_name:
            print(f"Step 2/2: Loading sampling model {args.sampling_model_name}...")
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()
            print(f"Sampling model loaded, device: {self.sampling_model.device}")
        print("[Initialization] FastDetectGPT ready\n")
        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'EleutherAI/gpt-neo-2.7B_EleutherAI/gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983,
                                                                'sigma1': 1.9935},
            'tiiuae/falcon-7b_tiiuae/falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
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
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer mismatch"
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
    """Process dataset in batches and return results"""
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

            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(texts)} samples")
        except Exception as e:
            print(f"Failed to process sample {i + 1}: {str(e)}")
            probs.append(np.nan)
            crits.append(np.nan)
            n_tokens_list.append(0)

    # Filter out invalid samples
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
    """Calculate evaluation metrics"""
    probs = results['probs']
    true_labels = results['true_labels']

    if len(np.unique(true_labels)) < 2:
        return {'error': 'Two classes required to calculate metrics'}

    # Binary classification predictions
    pred_labels = (probs >= 0.5).astype(int)

    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels),
        'roc_auc': roc_auc_score(true_labels, probs)
    }

    # ROC curve data
    fpr, tpr, _ = roc_curve(true_labels, probs)
    metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

    # PR curve data
    precision, recall, _ = precision_recall_curve(true_labels, probs)
    metrics['pr_curve'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'auc': auc(recall, precision)
    }

    return metrics


def plot_curves(metrics, output_dir='results'):
    """Plot ROC and PR curves"""
    os.makedirs(output_dir, exist_ok=True)

    # ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['roc_curve']['fpr'], metrics['roc_curve']['tpr'],
             label=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # PR curve
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['pr_curve']['recall'], metrics['pr_curve']['precision'],
             label=f"PR Curve (AUC = {metrics['pr_curve']['auc']:.4f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()


def plot_metrics_bar_chart(metrics, output_dir='results'):
    """Plot a bar chart of key metrics"""
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1', 0),
        metrics.get('roc_auc', 0)
    ]

    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.ylim(0, 1.1)  # Set y-axis limits to 0-1.1 to accommodate values
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'metrics_bar_chart.png'))
    plt.close()


def run_batch_evaluation(detector, texts, labels, args):
    """Run batch evaluation"""
    # Process data
    print(f"Starting processing of {len(texts)} samples...")
    results = evaluate_dataset(detector, texts, labels)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Save results
    output_dir = 'detectgpt_results'
    os.makedirs(output_dir, exist_ok=True)

    # Create serializable results dictionary
    serializable_results = {
        'probs': results['probs'].tolist(),  # Convert to Python list
        'crits': results['crits'].tolist(),  # Convert to Python list
        'true_labels': results['true_labels'].tolist(),  # Convert to Python list
        'avg_time': results['avg_time'],
        'total_samples': results['total_samples'],
        'valid_samples': results['valid_samples']
    }

    # Create serializable metrics dictionary
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            # Convert NumPy scalars to native Python types
            serializable_metrics[key] = value.item()
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            serializable_metrics[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    serializable_metrics[key][subkey] = subvalue.tolist()
                else:
                    serializable_metrics[key][subkey] = subvalue
        else:
            serializable_metrics[key] = value

    # Save raw results
    with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'results': serializable_results,
            'metrics': serializable_metrics
        }, f, indent=2, ensure_ascii=False)

    # Save evaluation report
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

    # Plot curves
    if 'error' not in metrics:
        plot_curves(metrics, output_dir)
        plot_metrics_bar_chart(metrics, output_dir)  # Add bar chart

    print("Evaluation completed. Results saved in directory:", output_dir)
    print("Key Metrics:")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
    print(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
    print(f"F1 Score: {metrics.get('f1', 'N/A'):.4f}")
    print(f"ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")


def load_preprocessed_data(input_file):
    """Load preprocessed data"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['texts'], data['labels']


def main():
    parser = argparse.ArgumentParser(description='Fast-DetectGPT Batch Detection Tool')
    parser.add_argument('--sampling_model_name', type=str, default="tiiuae/falcon-7b",
                        help='Sampling model name')
    parser.add_argument('--scoring_model_name', type=str, default="tiiuae/falcon-7b-instruct",
                        help='Scoring model name')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Compute device (cuda or cpu)')
    parser.add_argument('--cache_dir', type=str, default="../cache",
                        help='Model cache directory')
    parser.add_argument('--input_file', type=str, default="preprocessed_data.json",
                        help='Preprocessed data file path')

    args = parser.parse_args()

    # Print configuration
    print("=" * 50)
    print("Fast-DetectGPT Batch Detection Configuration")
    print("=" * 50)
    print(f"Sampling Model: {args.sampling_model_name}")
    print(f"Scoring Model: {args.scoring_model_name}")
    print(f"Compute Device: {args.device}")
    print(f"Input File: {args.input_file}")
    print("=" * 50)

    # 1. Initialize model
    print("=== Step 1: Model Initialization ===")
    detector = FastDetectGPT(args)

    # 2. Load preprocessed data
    print("\n=== Step 2: Load Preprocessed Data ===")
    texts, labels = load_preprocessed_data(args.input_file)
    print(f"Loaded: {len(texts)} texts ({sum(labels)} AI texts, {len(labels) - sum(labels)} human texts)")

    # 3. Run evaluation
    print("\n=== Step 3: Run Evaluation ===")
    run_batch_evaluation(detector, texts, labels, args)


if __name__ == '__main__':
    main()