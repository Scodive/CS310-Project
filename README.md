# 人类 vs AI 生成文本分类系统

这是一个基于BERT的文本分类系统，用于区分人类编写的文本和大型语言模型（LLM）生成的文本。系统支持中英文文本处理，并提供独立的训练脚本、评估功能和模型对比分析。

## 系统架构

### 主要组件

1.  **`text_classifier.py`**: 包含核心的文本分类逻辑。
    *   **`TextDataset`**: 自定义PyTorch数据集类，负责文本tokenization和数据预处理，将原始文本和标签转换为模型可接受的格式。
    *   **`BERTClassifier`**: 基于BERT的二分类模型架构，利用预训练的Transformer编码器进行特征提取，并通过一个分类头进行预测。
    *   **`TextClassificationTrainer`**: 核心训练器类，封装了模型的训练、验证、评估、保存和加载等功能。它支持多GPU训练 (使用 `nn.DataParallel`)，并能够从多种数据格式加载数据。

2.  **训练脚本**:
    *   **`train_chinese_model.py`**: 用于训练一个专门针对中文文本的分类模型 (默认使用 `bert-base-chinese`)。
    *   **`train_english_model.py`**: 用于训练一个专门针对英文文本的分类模型 (默认使用 `bert-base-uncased`)。
    *   **`train_bilingual_model.py`**: 用于训练一个能够同时处理中英文文本的分类模型 (默认使用 `bert-base-multilingual-cased`)。

3.  **模型对比脚本**:
    *   **`compare_trained_models.py`**: 用于加载已训练的多个模型，在标准化的测试集（中文、英文、混合）上进行评估，并生成包含性能指标（准确率、F1、AUC-ROC）、推理时间、GPU内存使用和参数量的对比报告和图表。

### 支持的预训练模型 (默认配置)

-   `bert-base-chinese`: 用于中文模型。
-   `bert-base-uncased`: 用于英文模型。
-   `bert-base-multilingual-cased`: 用于双语模型。
    (用户可以在训练脚本中更改这些默认设置)

## 安装要求

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
```
请确保您的PyTorch版本与CUDA版本兼容（如果使用GPU）。

## 数据格式

系统通过 `TextClassificationTrainer` 的 `load_data_from_specific_sources` 方法支持以下数据格式：

1.  **JSON 文件 (AI生成数据 - 格式1)**:
    *   文件结构: 一个JSON对象，包含一个名为 `"input"` 的键，其值为一个字典，该字典的每个值是AI生成的文本。
    *   示例: `{"input": {"0": "AI文本1...", "1": "AI文本2...", ...}}`
    *   `TextClassificationTrainer` 会自动提取 `"input"` 字典中的所有文本字符串。

2.  **JSON 文件 (人类书写/通用 - 格式2)**:
    *   文件结构: 一个JSON列表，列表中的每个元素是一个包含文本的字典。
    *   对于人类书写的数据，默认期望文本在 `"output"` 键下。
    *   示例 (人类): `[{"input": "...", "output": "人类文本1..."}, {"output": "人类文本2..."}]`
    *   对于AI数据，如果不是格式1，默认期望文本在 `"text"` 键下。
    *   示例 (AI通用): `[{"text": "AI文本1...", "label": 1}, {"text": "AI文本2...", "label": 1}]`
        (注意: `load_data_from_specific_sources` 会自动分配标签，所以JSON中的label字段会被忽略)

3.  **纯文本文件 (`.txt`)**:
    *   每个 `.txt` 文件包含一个完整的文本样本 (人类或AI)。
    *   文件名本身不用于区分标签；标签由文件所在的目录结构或传递给加载函数的文件列表决定。

在所有情况下，`load_data_from_specific_sources` 方法会根据传入的AI文件列表和人类文件列表自动为文本分配标签 (0 表示人类, 1 表示 AI)。

## 使用方法

### 1. 训练模型

首先，确保您的数据按照上述格式准备好，并放置在相应的路径下。然后，您可以运行以下任一脚本：

**训练中文模型:**
```bash
python train_chinese_model.py
```
-   默认数据路径:
    -   AI: `/home/hljiang/CS310-Natural_Language_Processing/generated/zh_qwen2/*.json`
    -   Human: `/home/hljiang/CS310-Natural_Language_Processing/human/zh_unicode/*.json`
-   模型保存路径: `./saved_models/chinese_model/`

**训练英文模型:**
```bash
python train_english_model.py
```
-   默认数据路径:
    -   AI: `/home/hljiang/CS310-Natural_Language_Processing/ghostbuster-data/essay/*/*.txt` (排除 `human` 目录)
    -   Human: `/home/hljiang/CS310-Natural_Language_Processing/ghostbuster-data/essay/human/*.txt`
-   模型保存路径: `./saved_models/english_model/`

**训练中英双语模型:**
```bash
python train_bilingual_model.py
```
-   默认数据路径: 合并上述中文和英文数据路径。
-   模型保存路径: `./saved_models/bilingual_model/`

**脚本参数修改**:
您可以直接修改这些训练脚本内部的变量来更改数据路径、模型名称、训练轮数、批次大小等参数。

### 2. 对比已训练的模型

在训练完一个或多个模型后，运行对比脚本：
```bash
python compare_trained_models.py
```
-   此脚本会自动加载 `./saved_models/` 下的 `chinese_model`, `english_model`, 和 `bilingual_model`。
-   它会在预定义的中文、英文和混合测试集上评估这些模型。
-   输出:
    -   `trained_models_comparison_report.json`: 包含详细指标的JSON报告。
    -   `trained_models_comparison_plots.png`: 可视化对比图表。

### 3. 直接使用 `TextClassificationTrainer` (高级)

您可以导入并直接使用 `TextClassificationTrainer` 类来进行更细致的控制或集成到其他工作流中。

```python
from text_classifier import TextClassificationTrainer, TextDataset
from torch.utils.data import DataLoader

# 初始化训练器
trainer = TextClassificationTrainer(
    model_name='bert-base-multilingual-cased', # 或其他模型
    max_length=512
)

# 假设 ai_files_list 和 human_files_list 是您的文件路径列表
texts, labels = trainer.load_data_from_specific_sources(
    ai_generated_files=ai_files_list,
    human_written_files=human_files_list
)

# _load_json_files_from_list 和 _load_plain_text_files_from_list 也可单独使用（但不推荐直接调用）

# 准备数据
# 注意: prepare_data 内部使用 train_test_split, 标签会自动用于分层抽样 (如果适用)
train_dataset, test_dataset = trainer.prepare_data(texts, labels, test_size=0.2)

# 训练模型 (如果GPU可用且多于一个，会自动使用DataParallel)
trainer.train(
    train_dataset=train_dataset,
    val_dataset=test_dataset, # 使用测试集作为验证集
    epochs=3,
    batch_size=16 # 总批次大小，DataParallel会自动分配到各GPU
)

# 评估模型
# evaluate 方法现在返回 (avg_loss, accuracy, detailed_metrics_dict)
final_test_loader = DataLoader(test_dataset, batch_size=16)
loss, acc, metrics = trainer.evaluate(final_test_loader)
print(f"Test Accuracy: {acc}, F1: {metrics['f1']}")

# 预测新文本
new_texts_to_predict = ["这是一个新的AI文本。", "This is a new human text."]
predictions, probabilities = trainer.predict(new_texts_to_predict)
for text, pred, prob in zip(new_texts_to_predict, predictions, probabilities):
    print(f'Text: {text} -> Prediction: {"AI" if pred == 1 else "Human"} (Prob AI: {prob[1]:.3f})')

# 保存和加载模型
trainer.save_model('./my_custom_model')

new_trainer = TextClassificationTrainer(model_name='bert-base-multilingual-cased') # 需要指定原始模型名
new_trainer.load_model('./my_custom_model')
predictions_loaded, _ = new_trainer.predict(["Another test."])
```

## 评估指标

系统和对比脚本会报告以下主要评估指标：

-   **Accuracy**: 准确率
-   **Precision**: 精确率 (针对AI类别)
-   **Recall**: 召回率 (针对AI类别)
-   **F1-Score**: F1分数 (针对AI类别)
-   **AUC-ROC**: ROC曲线下面积

对比脚本还会报告：
-   **参数量 (Parameters)**
-   **推理时间 (Inference Time)**
-   **峰值GPU内存使用 (Peak GPU Memory)**

## 输出文件

### 各训练脚本的输出:
-   模型文件:
    -   `./saved_models/{model_type}_model/model.pth`: 模型权重
    -   `./saved_models/{model_type}_model/config.json`: 模型配置
    -   `./saved_models/{model_type}_model/tokenizer/...`: Tokenizer文件
-   评估结果 (如果测试集非空):
    -   `./saved_models/{model_type}_model/evaluation_results.json`: 详细评估指标。
-   训练曲线图:
    -   `plots/training_curves.png` (注意：此文件会被后续训练覆盖，如需保留特定训练的曲线，请在运行后手动重命名或复制)

### `compare_trained_models.py` 的输出:
-   `trained_models_comparison_report.json`: JSON格式的详细对比报告。
-   `trained_models_comparison_plots.png`: 包含多个子图的PNG图像，可视化各模型的性能、参数、推理时间等。

## 预期结果的呈现

根据项目要求，您的目标可能是生成一个类似以下的2x2表格，比较不同方法（如监督学习 vs. 零样本）在不同语言上的表现。

| 方法       | 英文数据集指标 | 中文数据集指标 |
| :----------- | :------------- | :------------- |
| **监督学习模型 (本项目)** |                |                |
| `bert-base-uncased` (英) | (来自对比报告) | (不适用/低)    |
| `bert-base-chinese` (中) | (不适用/低)    | (来自对比报告) |
| `bert-base-multilingual` (双语) | (来自对比报告) | (来自对比报告) |
| **零样本方法** (由`zero_shot_detector.py`提供，若使用) | AA             | BB             |
| 其他方法...| CC             | DD             |

本项目中的 `compare_trained_models.py` 脚本主要关注已训练的监督学习模型的对比。您需要结合 `zero_shot_detector.py` (或其他零样本方法的脚本) 的结果来填充完整的表格。

## 数据集建议

### Ghostbuster English数据 (示例路径)
-   `/home/hljiang/CS310-Natural_Language_Processing/ghostbuster-data/essay/`
-   包含多个LLM（如claude等）生成的英文`.txt`文件，以及`human`子目录下的对应人类原文。

### 中文数据 (示例路径)
-   AI: `/home/hljiang/CS310-Natural_Language_Processing/generated/zh_qwen2/*.json`
-   Human: `/home/hljiang/CS310-Natural_Language_Processing/human/zh_unicode/*.json`
-   包含新闻、维基百科、网络小说等领域，AI部分由Qwen-2生成。

### 数据预处理建议
1.  **文本清理**: 确保文本内容干净，无过多噪声。
2.  **平衡数据集**: 虽然训练脚本会加载所有找到的文件，但在准备最终的训练语料时，注意类别的平衡可能有助于模型性能。
3.  **文本长度**: BERT模型有最大序列长度限制 (通常是512个token)。过长的文本会被截断。

## 注意事项

1.  **计算资源**: 训练BERT模型通常需要GPU。`TextClassificationTrainer` 支持多GPU训练 (通过 `nn.DataParallel`)。如果GPU内存不足，尝试减小 `batch_size` 或 `max_length` (在训练脚本中设置)。
2.  **训练时间**: 根据数据集大小、模型大小和硬件配置，训练可能需要几分钟到几小时不等。
3.  **过拟合**: 脚本中使用了训练集/测试集划分，并在训练时使用测试集作为验证集来监控性能。观察验证损失和准确率以避免严重过拟合。
4.  **模型下载**: 首次使用特定的Hugging Face模型时，系统会自动下载模型权重和tokenizer配置。确保网络连接通畅。后续运行会使用本地缓存。
5.  **字体**: `compare_trained_models.py` 脚本会尝试使用 `SimHei` 字体显示中文图表。如果系统中没有此字体，请安装或在脚本中修改为其他可用中文字体。

## 故障排除

### 常见问题
1.  **CUDA内存不足 (CUDA out of memory)**: 在对应的训练脚本中减小 `batch_size`。
2.  **模型下载问题**: 检查网络连接或配置Hugging Face的镜像源。
3.  **`FileNotFoundError`**: 仔细检查脚本中的数据路径是否正确指向您的数据文件。
4.  **`TypeError: ... got an unexpected keyword argument 'stratify'`**: 确保您使用的是最新版本的 `text_classifier.py` 和训练脚本，此问题已在先前版本中修复。
5.  **JSON解析错误 (lorsque vous traitez des fichiers .txt avec un chargeur JSON)**: 确保您使用的是最新版本的 `text_classifier.py` (特别是 `load_data_from_specific_sources` 方法)，它应该能正确处理混合文件类型。

## 扩展功能 (未来展望)

系统可以进一步扩展以支持：
1.  **更多预训练模型**: 集成如RoBERTa, XLNet, ELECTRA等其他Transformer架构。
2.  **更细致的零样本/少样本方法**: 实现更复杂的零样本检测逻辑。
3.  **用户界面**: 创建一个简单的Web界面进行文本输入和分类预测。
4.  **自动超参数调优**: 使用Optuna或Ray Tune等库进行超参数搜索。
