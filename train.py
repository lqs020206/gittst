import torch
import torch.nn as nn
import torch.optim as optim
from model import ParkingModel
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm  # Import tqdm
import matplotlib.pyplot as plt
# 超参数
batch_size = 200
learning_rate = 0.001
num_epochs_detection = 5
num_epochs_navigation = 5
motionDir = {
    0: [-1,  1],
    1: [ 0,  math.sqrt(2)],
    2: [ 1,  1],
    3: [-math.sqrt(2),  0],
    4: [ 0,  0],
    5: [ math.sqrt(2),  0],
    6: [-1, -1],
    7: [ 0, -math.sqrt(2)],
    8: [ 1, -1],
}
# 数据目录
data_train = "F:/Code/tingchechang/dataclassify1/train"  # 数据根目录
data_test = "F:/Code/tingchechang/dataclassify1/test"  # 数据根目录
# 数据转换
transform = transforms.Compose([
    transforms.Resize([360, 360]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def map_labels_to_positions(label1):
    positions = {
        0: 0,
        1000: 1,
        1100: 2,
        1110: 3,
    }
    positions_list = []
    label1 = torch.round(label1)
    for label in label1:
        label_int = int(torch.sum(label * torch.tensor([1000, 100, 10, 1])))
        position = positions[label_int]
        positions_list.append(position)
    return torch.tensor(positions_list).unsqueeze(1)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        for root, dirs, files in os.walk(self.root_dir):
            for filename in files:
                    #print(root)
                    labels = root[-6:].replace('\\', '')
                    # labels = os.path.basename(root).split("\\")[-2:]  # 获取路径中的标签信息，假设路径结构为"root_dir/类别1/类别2/文件.jpg"
                    #print(labels)
                    data.append((os.path.join(root, filename), labels))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, labels = self.data[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, labels
# 创建自定义数据加载器
train_dataset = CustomDataset(data_train, transform=transform)
test_dataset = CustomDataset(data_test, transform=transform)
# 创建数据加载器，用于车位检测模型
train_loader_detection = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 创建数据加载器，用于导航模型
#train_loader_navigation = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



# dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# # 打印图像和标签
# for batch in dataloader:
#     images, labels = batch
#     print(labels)




# 定义车位检测模型
parking_model = ParkingModel()
# pretrained_dict = torch.load("D:/OneDrive - sjtu.edu.cn/桌面/毕设代码/量化BSC/parking_spot_detection_model_epoch5.pth")
#
# # 应用预训练模型的参数到第一个全连接层及其之前的层
# parking_model.apply_pretrained(pretrained_dict)
criterion_detection = nn.CrossEntropyLoss()
optimizer_detection = optim.Adam(parking_model.parameters(), lr=learning_rate)



# 定义导航模型
# conditional_navigation_model = ConditionalNavigationModel(num_conditions=16)
criterion_navigation = nn.CrossEntropyLoss()
# optimizer_navigation = optim.Adam(conditional_navigation_model.parameters(), lr=learning_rate)
best_loss = float('inf')  # 初始化一个很大的损失值作为初始值
best_epoch = 0
acc_step = 0
epoch_losses = []
fig, ax = plt.subplots()
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
plt.ion()  # 开启交互模式

for epoch in range(num_epochs_detection):
    parking_model.train()
    running_loss = 0.0  # 用于累积每个 epoch 中损失值的变量
    train_loader_detection = tqdm(train_loader_detection, desc=f'Epoch {epoch + 1}/{num_epochs_detection}')
    for step, (images, labels) in enumerate(train_loader_detection, start=0):
        optimizer_detection.zero_grad()
        outputs, features = parking_model(images)
        # print(outputs.size())
        # 假设 outputs 的大小为 [batch_size, 14]，前四个是第一个四个输出
        first_four_outputs = outputs[:, :4]
        last_ten_outputs = outputs[:, -9:]
        # print(first_four_outputs.size())

        # first_four_labels = [torch.tensor([int(char) for char in label[:4]]) for label in labels]
        # last_ten_labels = [torch.tensor([int(char) for char in label[-1]]) for label in labels]
        first_four_labels = torch.tensor([int(char) for label in labels for char in label[:4]])
        #first_four_labels = first_four_labels.view(-1,4)
        first_four_labels = first_four_labels.float()
        first_four_outputs = first_four_outputs.reshape(-1)
        first_four_outputs = first_four_outputs.float()

        # probabilities = outputs[:, -10:]
        # # print(probabilities)
        # motion_direction_probs = F.softmax(probabilities, dim=1)
        # # print(motion_direction_probs)
        # max_probs_indices = torch.argmax(motion_direction_probs, dim=1)
        # print(max_probs_indices)
        # print(probabilities.size())
        # print(motion_direction_probs.size())
        # max_probs_indices_list = max_probs_indices.tolist()
        # max_probs_indices_list = [int(i) for i in max_probs_indices_list]


        last_ten_labels = torch.tensor([[int(char)] for label in labels for char in label[-1]])
        last_ten_labels = last_ten_labels.reshape(-1)

        last_ten_labels_list = last_ten_labels.tolist()
        last_ten_labels_list = [int(i) for i in last_ten_labels_list]
        # tuple_max=tuple(max_probs_indices_list)
        # tuple_labels=tuple(last_ten_labels_list)
        # probabilities = outputs[:, -10:]
        # motion_direction_probs = F.softmax(probabilities, dim=1)
        # max_probs_indices = torch.argmax(motion_direction_probs, dim=1)

        # print('outputs',first_four_outputs)
        # print('labels',first_four_labels)
        # print(last_ten_labels)
        # print(last_ten_outputs)
        # print(last_ten_labels)
        # 分别计算前四个和最后十个输出的损失
        # last_ten_labels = last_ten_labels.float()
        # max_probs_indices = max_probs_indices.float()
        # print(first_four_outputs)
        first_four_outputs = first_four_outputs.reshape(-1,4)
        first_four_labels = first_four_labels.reshape(-1,4)
        first_four_labels = map_labels_to_positions(first_four_labels)
        first_four_labels = first_four_labels.squeeze()

        # print(first_four_outputs.size())
        # print(first_four_labels.size())

        loss_first_four = criterion_detection(first_four_outputs, first_four_labels)
        # max_probs_indices_tensor = torch.tensor(max_probs_indices_list, dtype=torch.float32, requires_grad=True)
        last_ten_labels_tensor = torch.tensor(last_ten_labels_list, dtype=torch.float32, requires_grad=True)
        # print('max_probs_indices',max_probs_indices)
        # print('last_ten_labels',last_ten_labels)
        # threshold = 0.8  # 设置阈值
        # binary_first_four_outputs = (first_four_outputs >= threshold).int()
        # print('first_four_outputs',binary_first_four_outputs)
        # print('first_four_labels',first_four_labels)
        # print(last_ten_outputs)
        # print(last_ten_outputs.size())
        # print(last_ten_labels.size())
        loss_last_ten = criterion_navigation(last_ten_outputs, last_ten_labels)

        # loss_last_ten = criterion_navigation(tuple_max, tuple_labels)
        # print('first',loss_first_four)
        # print('last',loss_last_ten)
        # 合并损失（根据任务需求可能需要加权）
        # loss_last_ten = torch.tensor(loss_last_ten)
        loss = loss_first_four + loss_last_ten
        running_loss += loss.item()  # 累加每个 batch 的损失值
        # print('loss',loss)
        # 实时更新损失曲线
        # epoch_losses.append(loss.item())
        # ax.plot(epoch_losses, color='blue')
        # plt.pause(0.01)  # 添加小延迟，确保图形有足够的时间来更新
        loss.backward()
        optimizer_detection.step()

        train_loader_detection.set_description(f'Epoch {epoch + 1}/{num_epochs_detection}, Step {step + 1}/{len(train_loader_detection)}, Loss: {loss.item():.4f}')

        # Update tqdm progress bar
        train_loader_detection.update()
        # Append the current loss to the history
        if step % 50 == 49:  # 每隔一定步数输出一次
            parking_model.eval()  # 切换模型到评估模式
            with torch.no_grad():
                acc1_total_correct = 0
                acc2_total_correct = 0
                total_samples = 0

                for idx, (test_images, test_labels) in enumerate(test_loader):
                    test_outputs, _ = parking_model(test_images)



                    # 分离前四位和后十位
                    test_first_four_outputs = test_outputs[:, :4]
                    test_last_ten_outputs = test_outputs[:, -9:]

                    test_first_four_outputs = torch.tensor(test_first_four_outputs)
                    test_last_ten_outputs = torch.tensor(test_last_ten_outputs)
                    # threshold = 0.8  # 设置阈值
                    # binary_test_first_four_outputs = (test_first_four_outputs >= threshold).int()

                    # binary_test_first_four_outputs = binary_test_first_four_outputs.reshape(-1, 4)

                    test_first_four_labels = torch.tensor([[int(char)] for label in test_labels for char in label[:4]])
                    test_last_ten_labels = torch.tensor([[int(char)] for label in test_labels for char in label[-1]])

                    test_first_four_outputs = test_first_four_outputs.reshape(-1, 4)
                    test_first_four_labels = test_first_four_labels.reshape(-1, 4)
                    # print(binary_test_first_four_outputs.size())
                    # print(test_first_four_labels)
                    test_first_four_labels = map_labels_to_positions(test_first_four_labels)
                    correct_predictions1 = torch.eq(torch.max(test_first_four_outputs, dim=1)[1],
                                                    test_first_four_labels.squeeze()).sum().item()





                    # print('correct_predictions1',correct_predictions1)

                    correct_predictions2 = torch.eq(torch.max(test_last_ten_outputs, dim=1)[1],
                                                    test_last_ten_labels.squeeze()).sum().item()
                    # print('correct_predictions2',correct_predictions2)

                    acc1_total_correct += correct_predictions1
                    acc2_total_correct += correct_predictions2
                    total_samples += test_images.size(0)

                    # if (idx + 1) % 5 == 0:  # 每五个批次输出一次准确率
                    #     accuracy1 = acc1_total_correct / total_samples
                    #     accuracy2 = acc2_total_correct / total_samples

                    # print(f'Accuracy for first four digits: {accuracy1:.4f}')
                    # print(f'Accuracy for last ten digits: {accuracy2:.4f}')

                # 输出最终准确率
                final_accuracy1 = acc1_total_correct / total_samples
                final_accuracy2 = acc2_total_correct / total_samples
                print(f'Final Accuracy for first four digits: {final_accuracy1:.4f}')
                print(f'Final Accuracy for last ten digits: {final_accuracy2:.4f}')
                print('running_loss', running_loss / 50)
                running_loss = 0.0

            parking_model.train()
    torch.save(parking_model.state_dict(), f'model_epoch{epoch + 1}p0.05 bs100.pth')

    # print(f'Epoch {epoch + 1}/{num_epochs_detection}, Average Loss: {average_loss:.4f}')
    # acc_step += step  # 更新累计步数
print('finish training')


# 保存训练好的模型
torch.save(parking_model.state_dict(), 'parking_spot_detection_model11.pth')
train_loader_detection.close()

