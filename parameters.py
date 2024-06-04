import torch
import torch.nn as nn
from model import ParkingModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 定义你的神经网络模型
model = ParkingModel()

# 打印模型的参数量
print("模型的参数量：", count_parameters(model))
