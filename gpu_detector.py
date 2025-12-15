import torch

try:
    device_index = torch.cuda.current_device()
    print(f"当前使用的 GPU 设备索引为: {device_index}")
except RuntimeError:
    print("未检测到 GPU，正在使用 CPU。")