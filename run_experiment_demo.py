#!/usr/bin/env python3
"""
实验运行脚本：用于训练和评估文本分类模型
支持英文和中文数据集的监督学习方法
"""

import os
import json
import pandas as pd
import numpy as np
from text_classifier import TextClassificationTrainer
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self):
        self.results = {}
    
    def run_supervised_experiment(self, data_path, model_name, experiment_name, 
                                epochs=5, batch_size=16, max_length=512):
        """运行监督学习实验"""
        print(f"\n{'='*50}")
        print(f"开始实验: {experiment_name}")
        print(f"数据路径: {data_path}")
        print(f"模型: {model_name}")
        print(f"{'='*50}")
        
        # 初始化训练器
        trainer = TextClassificationTrainer(
            model_name=model_name,
            max_length=max_length
        )
        
        # 加载数据
        try:
            texts, labels = trainer.load_data(data_path)
            print(f"成功加载数据: {len(texts)} 个样本")
            print(f"标签分布: {pd.Series(labels).value_counts()}")
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
        
        # 准备数据
        train_dataset, test_dataset = trainer.prepare_data(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 创建验证集
        val_size = int(0.1 * len(texts))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        print(f"训练样本: {len(train_dataset)}")
        print(f"验证样本: {len(val_dataset)}")
        print(f"测试样本: {len(test_dataset)}")
        
        # 训练模型
        training_history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=2e-5
        )
        
        # 评估模型
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        test_loss, test_acc, detailed_metrics = trainer.evaluate(test_loader)
        
        # 保存模型
        model_save_path = f"./models/{experiment_name}"
        trainer.save_model(model_save_path)
        
        # 保存结果
        experiment_results = {
            'experiment_name': experiment_name,
            'model_name': model_name,
            'data_path': data_path,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'epochs': epochs,
            'batch_size': batch_size,
            'test_accuracy': detailed_metrics['accuracy'],
            'test_precision': detailed_metrics['precision'],
            'test_recall': detailed_metrics['recall'],
            'test_f1': detailed_metrics['f1'],
            'test_auc_roc': detailed_metrics['auc_roc'],
            'training_history': training_history
        }
        
        self.results[experiment_name] = experiment_results
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(
            detailed_metrics['labels'], 
            detailed_metrics['predictions'],
            experiment_name
        )
        
        return experiment_results
    
    def plot_confusion_matrix(self, y_true, y_pred, experiment_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.title(f'Confusion Matrix - {experiment_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{experiment_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_results(self):
        """比较不同实验的结果"""
        if not self.results:
            print("没有实验结果可比较")
            return
        
        # 创建结果对比表
        comparison_data = []
        for exp_name, results in self.results.items():
            comparison_data.append({
                'Experiment': exp_name,
                'Accuracy': f"{results['test_accuracy']:.4f}",
                'Precision': f"{results['test_precision']:.4f}",
                'Recall': f"{results['test_recall']:.4f}",
                'F1-Score': f"{results['test_f1']:.4f}",
                'AUC-ROC': f"{results['test_auc_roc']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + "="*80)
        print("实验结果对比")
        print("="*80)
        print(df_comparison.to_string(index=False))
        
        # 保存结果到文件
        df_comparison.to_csv('experiment_results_comparison.csv', index=False)
        
        # 绘制结果对比图
        self.plot_results_comparison(df_comparison)
    
    def plot_results_comparison(self, df_comparison):
        """绘制结果对比图"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = [float(val) for val in df_comparison[metric]]
                experiments = df_comparison['Experiment']
                
                axes[i].bar(experiments, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
                axes[i].set_title(f'{metric} 对比')
                axes[i].set_ylabel(metric)
                axes[i].set_ylim(0, 1)
                axes[i].tick_params(axis='x', rotation=45)
                
                # 在柱状图上显示数值
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 删除空的子图
        if len(metrics) < len(axes):
            for i in range(len(metrics), len(axes)):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_all_results(self, filename='all_experiment_results.json'):
        """保存所有实验结果"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        print(f"所有实验结果已保存到: {filename}")

def create_sample_datasets():
    """创建示例数据集用于测试"""
    
    # 英文数据集
    english_data = {
        'text': [
            "The weather is beautiful today and I feel happy.",
            "This research paper presents novel findings in the field.",
            "I love spending time with my family on weekends.",
            "The company announced record profits this quarter.",
            "Generated text often lacks the natural flow of human writing.",
            "According to the analysis, the results demonstrate significant improvements.",
            "The implementation of this methodology yields optimal performance outcomes.",
            "Based on comprehensive evaluation, the proposed approach shows effectiveness."
        ],
        'label': [0, 0, 0, 0, 1, 1, 1, 1]  # 0: human, 1: AI
    }
    
    # 中文数据集
    chinese_data = {
        'text': [
            "今天天气真好，我感到很开心。",
            "这篇论文在该领域提出了新颖的发现。",
            "我喜欢在周末和家人度过时光。",
            "公司宣布了本季度创纪录的利润。",
            "生成的文本通常缺乏人类写作的自然流畅性。",
            "根据分析结果显示，该方法取得了显著的改进效果。",
            "通过实施这种方法论，可以获得最优的性能结果。",
            "基于综合评估，所提出的方法显示出了有效性。"
        ],
        'label': [0, 0, 0, 0, 1, 1, 1, 1]  # 0: human, 1: AI
    }
    
    # 保存数据集
    os.makedirs('data', exist_ok=True)
    
    pd.DataFrame(english_data).to_csv('data/english_sample.csv', index=False)
    pd.DataFrame(chinese_data).to_csv('data/chinese_sample.csv', index=False)
    
    print("示例数据集已创建:")
    print("- data/english_sample.csv")
    print("- data/chinese_sample.csv")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行文本分类实验')
    parser.add_argument('--create_sample', action='store_true', 
                       help='创建示例数据集')
    parser.add_argument('--english_data', type=str, 
                       default='data/english_sample.csv',
                       help='英文数据集路径')
    parser.add_argument('--chinese_data', type=str,
                       default='data/chinese_sample.csv',
                       help='中文数据集路径')
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 创建示例数据集（如果需要）
    if args.create_sample:
        create_sample_datasets()
    
    # 创建实验运行器
    runner = ExperimentRunner()
    
    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    
    # 实验配置
    experiments = [
        {
            'name': 'English_BERT_Multilingual',
            'data_path': args.english_data,
            'model_name': 'bert-base-multilingual-cased'
        },
        {
            'name': 'Chinese_BERT_Multilingual',
            'data_path': args.chinese_data,
            'model_name': 'bert-base-multilingual-cased'
        },
        {
            'name': 'English_BERT_Base',
            'data_path': args.english_data,
            'model_name': 'bert-base-uncased'
        },
        {
            'name': 'Chinese_BERT_Chinese',
            'data_path': args.chinese_data,
            'model_name': 'bert-base-chinese'
        }
    ]
    
    # 运行实验
    for exp in experiments:
        try:
            result = runner.run_supervised_experiment(
                data_path=exp['data_path'],
                model_name=exp['model_name'],
                experiment_name=exp['name'],
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            if result:
                print(f"实验 {exp['name']} 完成")
            else:
                print(f"实验 {exp['name']} 失败")
        except Exception as e:
            print(f"实验 {exp['name']} 出现错误: {e}")
    
    # 比较结果
    runner.compare_results()
    
    # 保存所有结果
    runner.save_all_results()
    
    print("\n所有实验完成！")

if __name__ == "__main__":
    main() 