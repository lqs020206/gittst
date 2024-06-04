import torch.nn as nn
import torch.optim as optim
from model import ParkingModel
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2
src_pts = np.array([
    [364, 100],
    [121, 893],
    [1713, 908],
    [1500, 125]
], dtype=np.float32)
dst_pts = np.array([
    [0, 0],
    [0, 360],
    [360, 360],
    [360, 0]
], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
def test_image(image_path, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将numpy数组转换为PIL图像
        transforms.Resize([360, 360]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 读取图像并进行预处理
    image = Image.open(image_path)
    image = np.array(image)
    image = cv2.warpPerspective(image, M, (360, 360))
    image = transform(image).unsqueeze(0)  # 增加一个维度以符合模型输入的形状
    # 设置模型为评估模式，关闭梯度计算
    model.eval()
    with torch.no_grad():
        # 将图像输入模型并获取输出
        output, features = model(image)

    return output, features

# 加载已训练的模型
model = ParkingModel()
model.load_state_dict(torch.load('model_epoch5p0.05 bs100.pth'))

# 测试图像的文件路径
# test_image_path = "E:/Downloads/192.168.139.1_01_20240223155008505.jpeg"
import os

# # 指定文件夹路径
folder_path = "E:/Downloads/"
#
# # 获取文件夹中所有文件的路径
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 获取最新文件的路径
latest_file = max(all_files, key=os.path.getctime)
#
# # 使用最新文件的路径作为测试图片路径
test_image_path = latest_file
# test_image_path = "F:/Code/tingchechang/datayoloclassify/1000/3/2024-01-06_18-29-13-017379.jpg"
# 测试图像并获取模型输出和特征图
output, features = test_image(test_image_path, model)
threshold = 0.8  # 设置阈值

parking_info = (output[:, :4] >= threshold).int()

# 对后十位选择最大概率的类别
motion_probs = F.softmax(output[:, 4:], dim=1)
max_probs_indices = torch.argmax(motion_probs, dim=1)

# 输出结果
print("Parking Information:", parking_info)
print("Motion Direction:", max_probs_indices)


# 将每层特征图放在同一个图里
fig, axs = plt.subplots(1, len(features) + 1, figsize=(15, 3))

# 显示输入图像
# input_image = torchvision.utils.make_grid(image)
# axs[0].imshow(input_image.permute(1, 2, 0).detach().numpy())
# axs[0].set_title('Input Image')

# 显示每层特征图
layer_labels = ["Conv1", "ReLU1", "MaxPool1", "Conv2", "ReLU2", "MaxPool2", "Conv3", "ReLU3", "MaxPool3", "Conv4", "ReLU4", "MaxPool4"]

for idx, feature_map in enumerate(features):
    if len(feature_map.shape) == 4:  # 检查是否是四维张量
        num_channels = feature_map.size(1)
        grid_size = math.ceil(math.sqrt(num_channels))  # 计算每个卷积层的特征图在子图中的排列大小
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle(layer_labels[idx], fontsize=16)  # 添加卷积层标签

        for i in range(num_channels):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].imshow(feature_map[0, i].cpu().detach().numpy(), cmap='viridis')
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Channel {i}')

        plt.show()
