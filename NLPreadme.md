中文模型：

train_chinese_model.py

Epoch 1: Train Loss: 0.0327, Val Loss: 0.0136, Val Acc: 0.9965
训练 Epoch 2/3: 100%|███████████████████████| 1500/1500 [05:40<00:00,  4.41it/s, loss=2.54e-5]
Epoch 2: Train Loss: 0.0069, Val Loss: 0.0131, Val Acc: 0.9973
训练 Epoch 3/3: 100%|███████████████████████| 1500/1500 [05:40<00:00,  4.41it/s, loss=1.55e-5]
Epoch 3: Train Loss: 0.0014, Val Loss: 0.0183, Val Acc: 0.9977
模型训练完成。
开始评估模型...
评估中: 100%|███████████████████████████████████████████████| 375/375 [00:32<00:00, 11.64it/s]

评估结果:
Loss: 0.0183
Accuracy: 0.9977
Precision: 0.9963
Recall: 0.9990
F1-Score: 0.9977
AUC-ROC: 0.9997



英文模型：

train_english_model.py

开始训练...
训练 Epoch 1/3: 100%|████████████████████████| 800/800 [01:43<00:00,  7.70it/s, loss=0.000334]
Epoch 1: Train Loss: 0.0708, Val Loss: 0.0695, Val Acc: 0.9894
训练 Epoch 2/3: 100%|█████████████████████████| 800/800 [01:43<00:00,  7.71it/s, loss=0.00021]
Epoch 2: Train Loss: 0.0107, Val Loss: 0.0273, Val Acc: 0.9956
训练 Epoch 3/3: 100%|█████████████████████████| 800/800 [01:43<00:00,  7.70it/s, loss=4.06e-5]
Epoch 3: Train Loss: 0.0027, Val Loss: 0.0483, Val Acc: 0.9906
模型训练完成。
开始评估模型...
评估中: 100%|███████████████████████████████████████████████| 200/200 [00:10<00:00, 19.74it/s]

评估结果:
Loss: 0.0483
Accuracy: 0.9906
Precision: 0.9901
Recall: 0.9993
F1-Score: 0.9947
AUC-ROC: 0.9998



双语模型

train_bilingual_model.py

<img width="686" alt="WechatIMG25374" src="https://github.com/user-attachments/assets/89e56e08-53af-4947-bd75-7d8996cf7cf5" />


检测到 8 个GPU。使用 nn.DataParallel 进行训练。
开始训练...
训练 Epoch 1/3: 100%|██████████████████████████| 119/119 [03:18<00:00,  1.67s/it, loss=0.0272]
Epoch 1: Train Loss: 0.1290, Val Loss: 0.0180, Val Acc: 0.9945
训练 Epoch 2/3: 100%|█████████████████████████| 119/119 [03:19<00:00,  1.67s/it, loss=0.00395]
Epoch 2: Train Loss: 0.0133, Val Loss: 0.0141, Val Acc: 0.9957
训练 Epoch 3/3: 100%|█████████████████████████| 119/119 [03:18<00:00,  1.67s/it, loss=0.00168]
Epoch 3: Train Loss: 0.0048, Val Loss: 0.0136, Val Acc: 0.9962
模型训练完成。
开始评估模型...
评估中: 100%|███████████████████████████████████████████████| 475/475 [04:10<00:00,  1.90it/s]

评估结果:
Loss: 0.0135
Accuracy: 0.9962
Precision: 0.9950
Recall: 0.9984
F1-Score: 0.9967
AUC-ROC: 0.9999
