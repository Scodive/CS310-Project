import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    """自定义数据集类，用于处理文本分类数据"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用tokenizer编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier(nn.Module):
    """基于BERT的二分类模型"""
    
    def __init__(self, model_name, num_classes=2, dropout_rate=0.3):
        super(BERTClassifier, self).__init__()
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS] token的输出进行分类
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class TextClassificationTrainer:
    """文本分类训练器"""
    
    def __init__(self, model_name='bert-base-multilingual-cased', max_length=512, device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
        print(f"使用设备: {self.device}")
        print(f"使用模型: {model_name}")
    
    def load_data(self, data_path, text_column='text', label_column='label'):
        """加载数据"""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("支持的文件格式: .json, .csv")
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        return texts, labels
    
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """准备训练和测试数据"""
        # 分割数据
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # 创建数据集
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        test_dataset = TextDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        
        return train_dataset, test_dataset
    
    def train(self, train_dataset, val_dataset=None, epochs=3, batch_size=16, 
              learning_rate=2e-5, warmup_steps=0, weight_decay=0.01):
        """训练模型"""
        
        # 初始化模型
        self.model = BERTClassifier(self.model_name).to(self.device)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        # 优化器和调度器
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练历史记录
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        print("开始训练...")
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch+1}/{epochs}')
            
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, criterion, verbose=False)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}')
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses, val_accuracies)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def evaluate(self, dataloader, criterion=None, verbose=True):
        """评估模型"""
        if not self.model:
            raise ValueError("模型未训练，请先调用train()方法")
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            eval_pbar = tqdm(dataloader, desc='评估中') if verbose else dataloader
            
            for batch in eval_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # 获取预测结果和概率
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 正类概率
        
        avg_loss = total_loss / len(dataloader)
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        try:
            auc_roc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc_roc = 0.0
        
        if verbose:
            print(f"\n评估结果:")
            print(f"Loss: {avg_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC-ROC: {auc_roc:.4f}")
        
        return avg_loss, accuracy, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def predict(self, texts, batch_size=16):
        """对新文本进行预测"""
        if not self.model:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 创建临时数据集（标签为0，不会被使用）
        temp_labels = [0] * len(texts)
        temp_dataset = TextDataset(texts, temp_labels, self.tokenizer, self.max_length)
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size)
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(temp_loader, desc='预测中'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_predictions, all_probabilities
    
    def save_model(self, save_path):
        """保存模型"""
        if not self.model:
            raise ValueError("模型未训练，请先调用train()方法")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pth'))
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # 保存模型配置
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length
        }
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path):
        """加载模型"""
        # 加载配置
        with open(os.path.join(load_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.max_length = config['max_length']
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        # 加载模型
        self.model = BERTClassifier(self.model_name).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(load_path, 'model.pth'), map_location=self.device))
        
        print(f"模型已从 {load_path} 加载")
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        """绘制训练曲线"""
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='训练损失')
        if val_losses:
            plt.plot(epochs, val_losses, 'r-', label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        if val_accuracies:
            plt.plot(epochs, val_accuracies, 'g-', label='验证准确率')
            plt.title('验证准确率')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_sample_data():
    """创建示例数据用于测试"""
    human_texts = [
        "这是一篇人类写的新闻报道，内容真实可信。",
        "The author carefully crafted this sentence with deliberate word choices.",
        "人工智能技术在近年来取得了显著进展，但仍面临诸多挑战。",
        "This novel explores the complex relationships between characters in a realistic manner."
    ]
    
    ai_texts = [
        "根据最新数据显示，该项技术具有广阔的应用前景和发展潜力。",
        "The implementation of advanced algorithms facilitates optimal performance metrics.",
        "综合考虑各种因素，我们可以得出以下结论和建议。",
        "This comprehensive analysis demonstrates the effectiveness of the proposed methodology."
    ]
    
    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0: human, 1: AI
    
    return texts, labels

# 使用示例
def main():
    """主函数，演示如何使用文本分类器"""
    # 创建训练器实例
    trainer = TextClassificationTrainer(
        model_name='bert-base-multilingual-cased',  # 支持中英文
        max_length=512
    )
    
    # 创建示例数据
    texts, labels = create_sample_data()
    
    # 准备数据
    train_dataset, test_dataset = trainer.prepare_data(texts, labels, test_size=0.3)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    # 训练模型
    training_history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        epochs=2,  # 示例中使用较少的epoch
        batch_size=2,  # 小批次用于演示
        learning_rate=2e-5
    )
    
    # 评估模型
    print("\n=== 测试集评估结果 ===")
    test_loss, test_acc, detailed_metrics = trainer.evaluate(
        DataLoader(test_dataset, batch_size=2)
    )
    
    # 保存模型
    trainer.save_model('./saved_model')
    
    # 预测新文本
    new_texts = [
        "这是一个测试文本，用于验证模型的预测能力。",
        "This is a test sentence to verify the model's prediction capability."
    ]
    
    predictions, probabilities = trainer.predict(new_texts)
    
    print("\n=== 预测结果 ===")
    for i, (text, pred, prob) in enumerate(zip(new_texts, predictions, probabilities)):
        print(f"文本 {i+1}: {pred} (Human概率: {prob[0]:.3f}, AI概率: {prob[1]:.3f})")
        print(f"内容: {text}\n")

if __name__ == "__main__":
    main()