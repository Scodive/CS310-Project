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
import glob
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
        self.is_multi_gpu = False # 初始化多GPU标志
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        
        print(f"使用设备: {self.device}")
        print(f"使用模型: {model_name}")
    
    def _load_json_files_from_list(self, file_paths, text_key='text'):
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
                            elif isinstance(item, str): # 处理字符串列表: ["text1", "text2"]
                                current_file_texts.append(item)
                    # 处理单个JSON对象，其中包含指定的text_key: {text_key: "value"}
                    elif isinstance(data, dict) and text_key in data and isinstance(data[text_key], str):
                        current_file_texts.append(data[text_key])
                
                if current_file_texts:
                    all_texts.extend(current_file_texts)
                    print(f"成功加载: {file_path}, 提取到 {len(current_file_texts)} 条文本 (此文件), {len(all_texts)} 条文本 (累计)")
                else:
                    # 只有在真正没有提取到文本时才打印警告，避免在文件类型不匹配时（例如空文件或非预期结构）误报
                    if os.path.getsize(file_path) > 0 : # 仅当文件非空但未提取到内容时警告
                         print(f"警告: 未从 {file_path} 提取到任何文本。请检查文件结构和 text_key (如果适用)。")

            except json.JSONDecodeError as e:
                print(f"加载或解析JSON文件 {file_path} 失败: JSON格式错误 - {e}")
            except Exception as e:
                print(f"加载或解析文件 {file_path} 时发生未知错误: {e}")
        return all_texts

    def _load_plain_text_files_from_list(self, file_paths):
        """从纯文本文件路径列表中加载文本"""
        all_texts = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip(): #确保内容不为空
                        all_texts.append(content.strip())
                print(f"成功加载: {file_path}, 提取到1条文本 (此文件), {len(all_texts)} 条文本 (累计)")
            except FileNotFoundError:
                print(f"错误: 文件未找到 {file_path}")
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {e}")
        return all_texts

    def load_data_from_specific_sources(self, ai_generated_files, human_written_files):
        """从指定的AI和人类来源加载数据并自动打标签。
        可以处理 .json 和 .txt 文件，即使在同一个列表中混合存在。
        """
        ai_texts_final = []
        if ai_generated_files:
            ai_json_files = [f for f in ai_generated_files if f.lower().endswith('.json')]
            ai_txt_files = [f for f in ai_generated_files if f.lower().endswith('.txt')]
            ai_unknown_files = [f for f in ai_generated_files if not f.lower().endswith('.json') and not f.lower().endswith('.txt')]

            if ai_json_files:
                print(f"检测到 {len(ai_json_files)} 个AI JSON文件，使用JSON加载器。")
                ai_texts_final.extend(self._load_json_files_from_list(ai_json_files, text_key='text'))
            if ai_txt_files:
                print(f"检测到 {len(ai_txt_files)} 个AI TXT文件，使用纯文本加载器。")
                ai_texts_final.extend(self._load_plain_text_files_from_list(ai_txt_files))
            if ai_unknown_files:
                for f_unknown in ai_unknown_files:
                    print(f"警告: AI文件 {f_unknown} 的格式未知，已跳过。")
        
        human_texts_final = []
        if human_written_files:
            human_json_files = [f for f in human_written_files if f.lower().endswith('.json')]
            human_txt_files = [f for f in human_written_files if f.lower().endswith('.txt')]
            human_unknown_files = [f for f in human_written_files if not f.lower().endswith('.json') and not f.lower().endswith('.txt')]

            if human_json_files:
                print(f"检测到 {len(human_json_files)} 个人类JSON文件，使用JSON加载器 (text_key='output')。")
                human_texts_final.extend(self._load_json_files_from_list(human_json_files, text_key='output'))
            if human_txt_files:
                print(f"检测到 {len(human_txt_files)} 个人类TXT文件，使用纯文本加载器。")
                human_texts_final.extend(self._load_plain_text_files_from_list(human_txt_files))
            if human_unknown_files:
                for f_unknown in human_unknown_files:
                    print(f"警告: Human文件 {f_unknown} 的格式未知，已跳过。")

        if not ai_texts_final and not human_texts_final:
            raise ValueError("未能从任何来源加载到数据。请检查文件路径和格式。")
            
        texts = human_texts_final + ai_texts_final
        labels = [0] * len(human_texts_final) + [1] * len(ai_texts_final) # 0 for human, 1 for AI
        
        if not texts:
             raise ValueError("未能从指定文件中提取任何文本。请确保text_key正确且文件内容有效。")

        return texts, labels

    def load_data(self, data_path, text_column='text', label_column='label'):
        """加载数据 (保留原始方法，用于单个CSV/JSON文件)"""
        if data_path.endswith('.json'):
            texts = []
            labels = []
            with open(data_path, 'r', encoding='utf-8') as f:
                # 尝试按JSON Lines (jsonl) 格式读取
                try:
                    for line in f:
                        item = json.loads(line)
                        if isinstance(item, dict) and text_column in item and label_column in item:
                            texts.append(item[text_column])
                            labels.append(item[label_column])
                    if not texts: # 如果jsonl读取失败，尝试作为普通json列表读取
                        f.seek(0) # 重置文件指针
                        data_list = json.load(f)
                        if isinstance(data_list, list):
                             for item in data_list:
                                if isinstance(item, dict) and text_column in item and label_column in item:
                                    texts.append(item[text_column])
                                    labels.append(item[label_column])
                except json.JSONDecodeError: # 如果jsonl读取失败，尝试作为普通json列表读取
                    f.seek(0) # 重置文件指针
                    data_list = json.load(f)
                    if isinstance(data_list, list):
                            for item in data_list:
                                if isinstance(item, dict) and text_column in item and label_column in item:
                                    texts.append(item[text_column])
                                    labels.append(item[label_column])
            if not texts:
                 raise ValueError(f"无法从JSON文件 {data_path} 中正确解析数据。请确保文件格式正确 (JSON列表或JSON Lines) 且包含 '{text_column}' 和 '{label_column}' 键。")

        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            texts = df[text_column].tolist()
            labels = df[label_column].tolist()
        else:
            raise ValueError("支持的文件格式: .json, .csv (用于单一文件加载)")
        
        return texts, labels
    
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """准备训练和测试数据"""
        
        stratify_param = None
        if labels and len(set(labels)) >= 2:
            stratify_param = labels
        elif labels: # 意味着 labels 存在但只有一个类别
            print("警告: 数据中只存在一种标签，无法进行分层抽样。将使用普通抽样进行数据分割。")
        # 如果 labels 为空或None, stratify_param 保持 None, train_test_split 会处理

        # 分割数据
        try:
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=test_size, random_state=random_state, stratify=stratify_param
            )
        except ValueError as e:
            # 如果即使 stratify_param 为 None（例如，由于样本总数太少无法分割），仍然出错
            print(f"分割数据时发生错误 (可能由于样本太少): {e}. 尝试不使用分层抽样（如果之前尝试了）。")
            # 再次尝试，明确不使用分层，以防 stratify_param 由于某种原因不是 None 但仍然失败
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=test_size, random_state=random_state, stratify=None
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
        self.is_multi_gpu = False # 重置/设置 multi_gpu 标志
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"检测到 {torch.cuda.device_count()} 个GPU。使用 nn.DataParallel 进行训练。")
            self.model = nn.DataParallel(self.model)
            self.is_multi_gpu = True
        elif torch.cuda.is_available():
            print(f"检测到 1 个GPU。")
        else:
            print("未检测到GPU，使用CPU进行训练。")

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
                # 确保 evaluate 返回两个值（如果verbose=False）
                eval_output = self.evaluate(val_loader, criterion, verbose=False)
                if isinstance(eval_output, tuple) and len(eval_output) == 3: # 包含detailed_metrics
                     val_loss, val_acc, _ = eval_output
                elif isinstance(eval_output, tuple) and len(eval_output) == 2: # 不包含detailed_metrics
                     val_loss, val_acc = eval_output
                else:
                    raise ValueError(f"evaluate返回了意外的输出: {eval_output}")

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
        
        effective_criterion = criterion if criterion else nn.CrossEntropyLoss()
        
        with torch.no_grad():
            eval_pbar = tqdm(dataloader, desc='评估中') if verbose else dataloader
            
            for batch in eval_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = effective_criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # 获取预测结果和概率
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 正类概率
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions) if all_labels else 0
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', zero_division=0
        ) if all_labels else (0,0,0,None)
        
        try:
            auc_roc = roc_auc_score(all_labels, all_probabilities) if all_labels and all_probabilities else 0
        except ValueError: # Catches "Only one class present in y_true. ROC AUC score is not defined in that case."
            auc_roc = 0.0
        
        detailed_metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }

        if verbose:
            print(f"\n评估结果:")
            print(f"Loss: {avg_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC-ROC: {auc_roc:.4f}")
            return avg_loss, accuracy, detailed_metrics_dict
        else:
            # 当 verbose=False 时，也返回详细指标字典，
            # 调用方 (如训练循环或对比脚本) 可以选择使用哪些部分。
            return avg_loss, accuracy, detailed_metrics_dict

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
        model_state_to_save = self.model.module.state_dict() if self.is_multi_gpu else self.model.state_dict()
        torch.save(model_state_to_save, os.path.join(save_path, 'model.pth'))
        
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
        config_path = os.path.join(load_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.max_length = config['max_length']
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        # 初始化基础模型并移动到设备
        # 注意：不要在这里直接用DataParallel包装，先加载state_dict
        base_model = BERTClassifier(self.model_name).to(self.device)
        
        model_path = os.path.join(load_path, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        # 加载状态字典
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 如果保存的是 DataParallel 包装的模型 (module.state_dict)，
        # 而当前环境是非DataParallel，或者要加载到DataParallel，通常直接加载也可以，
        # 但有时可能需要调整键名 (如果state_dict的键以'module.'开头而模型不是DataParallel时)。
        # PyTorch的load_state_dict通常能处理好 'module.' 前缀。
        base_model.load_state_dict(state_dict)
        
        self.model = base_model # 先赋值基础模型
        self.is_multi_gpu = False # 重置标志

        # 如果当前环境有多个GPU，则包装模型
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"加载后检测到 {torch.cuda.device_count()} 个GPU。使用 nn.DataParallel。")
            self.model = nn.DataParallel(self.model) # self.model 现在是 base_model
            self.is_multi_gpu = True
        elif torch.cuda.is_available():
             print(f"加载后检测到 1 个GPU。")
        else:
            print("加载后未检测到GPU，模型将在CPU上运行。")
        
        print(f"模型已从 {load_path} 加载")
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        """绘制训练曲线"""
        epochs_range = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, 'b-', label='训练损失')
        if val_losses:
            plt.plot(epochs_range, val_losses, 'r-', label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        if val_accuracies: # 只有在有验证准确率时才绘制第二个子图
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, val_accuracies, 'g-', label='验证准确率')
            plt.title('验证准确率')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        # 确保目录存在
        os.makedirs("plots", exist_ok=True)
        plt.savefig('plots/training_curves.png', dpi=300, bbox_inches='tight')
        # plt.show() # 在脚本模式下通常不调用show()，除非需要交互式显示

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
        val_dataset=test_dataset, # 将test_dataset作为验证集传入
        epochs=2,  # 示例中使用较少的epoch
        batch_size=2,  # 小批次用于演示
        learning_rate=2e-5
    )
    
    # 评估模型
    print("\n=== 测试集评估结果 ===")
    # 创建一个新的 DataLoader 用于最终测试评估
    final_test_loader = DataLoader(test_dataset, batch_size=2)
    test_loss, test_acc, detailed_metrics = trainer.evaluate(
        final_test_loader # 使用独立的 DataLoader
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
    for i, (text, pred, prob) in enumerate(zip(new_texts, predictions, probabilities)):\
        # 确保 prob 是一个列表或元组，并且至少有两个元素
        if isinstance(prob, (list, tuple, np.ndarray)) and len(prob) >= 2:
            print(f"文本 {i+1}: {pred} (Human概率: {prob[0]:.3f}, AI概率: {prob[1]:.3f})")
        else: # 如果prob格式不符合预期，打印原始prob值
            print(f"文本 {i+1}: {pred} (Probabilities: {prob})")
        print(f"内容: {text}\n")

if __name__ == "__main__":
    main() 