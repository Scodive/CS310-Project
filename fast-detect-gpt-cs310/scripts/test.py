import torch
print(torch.cuda.is_available())  # 应为 True
print(torch.cuda.device_count())   # 应 > 0