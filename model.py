import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def Quant(Vx, Q, RQM):
    return np.round(Q * Vx) - RQM


def QuantRevert(VxQuant, Q, RQM):
    return (VxQuant + RQM) / Q


def QuantToBinary(VxQuant, quant_bits):
    binary_array = np.unpackbits(VxQuant.astype(np.uint8), axis=-1)
    return binary_array[:, -quant_bits:]


def ListQuant(data_list, quant_bits):
    # Convert data to NumPy array
    data_array = np.array(data_list)

    # Compute data range
    data_min = np.min(data_array)
    data_max = np.max(data_array)

    # Compute quantization parameters
    Q = ((1 << quant_bits) - 1) * 1.0 / (data_max - data_min)
    RQM = int(round(Q * data_min))

    # Quantize the data
    quant_data = Quant(data_array, Q, RQM)

    # Return quantization parameters and quantized data
    return Q, RQM, quant_data


def BinaryToDecimal(binary_array, Q, RQM):
    decimal_array = np.dot(binary_array, 1 << np.arange(binary_array.shape[-1] - 1, -1, -1))
    return QuantRevert(decimal_array, Q, RQM)


def ListQuantRevert(quant_list, Q, RQM):
    quant_revert_data_list = QuantRevert(quant_list, Q, RQM)
    return quant_revert_data_list


# def AWGN_channel(input_signal, SNR_dB):
#     # 计算信号的功率
#     signal_power = np.mean(np.abs(input_signal)**2)
#
#     # 根据给定的信噪比（SNR）计算噪声功率
#     noise_power = signal_power / (10**(SNR_dB / 10))
#
#     # 生成高斯白噪声并进行缩放
#     noise = np.sqrt(noise_power) * np.random.randn(*input_signal.shape)
#     # print("noise",noise)
#
#     # 将噪声叠加到输入信号上
#     output_signal = input_signal + noise
#
#     return output_signal
def AWGN_channel(input_signal, SNR_dB):
    # 计算信号的功率
    signal_power = torch.mean(torch.abs(input_signal)**2)

    # 根据给定的信噪比（SNR）计算噪声功率
    noise_power = signal_power / (10**(SNR_dB / 10))

    # 生成高斯白噪声并进行缩放
    noise = torch.sqrt(noise_power) * torch.randn_like(input_signal)

    # 将噪声叠加到输入信号上
    output_signal = input_signal + noise

    return output_signal
class ParkingModel(nn.Module):
    """
    接受摄像头传来的图像
    输入尺寸： 3x360x360 （假设图像尺寸为3通道，360x360像素）
    输出：4位二进制数，表示车位情况
    """
    def __init__(self):
        super(ParkingModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(24, 36, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(36, 48, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2,padding=1),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.fc1 = nn.Linear(64*20*20, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.dropout = nn.Dropout(0.1)  # 添加 dropout
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 13)

    def forward(self, x):
        features = []  # 存储每层特征图

        # 卷积层
        for layer in self.cnn:
            x = F.relu(layer(x))
            features.append(x.clone().detach())

        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        # features.append(x.clone().detach())
        # Q, RQM, quant_data = ListQuant(x.tolist(), 8)  # 返回单个量化数据

        # outputdata = AWGN_channel(quant_data, 5)

        # x = ListQuantRevert(outputdata, Q, RQM)
        # print(x)

        # x = torch.tensor(x, dtype=torch.float32)
        # x = x.reshape(-1, 120)



        x = F.relu(self.fc2(x))
        # print(x)
        # x =  x.view(-1)
        # # print(x.size())
        # # 进行量化

        # print(x.size())
        # x = torch.tensor(quant_binary, dtype=torch.float32)  # Convert NumPy array back to tensor
        # print("Quantized output:", x)  # Print the quantized output

        x = F.relu(self.fc3(x))
        # features.append(x.clone().detach())
        # x = AWGN_channel(x, 0)
        # Q, RQM, quant_data = ListQuant(x.tolist(), 8)  # 返回单个量化数据
        #
        # x = ListQuantRevert(quant_data, Q, RQM)
        # # print(x)
        #
        # x = torch.tensor(x, dtype=torch.float32)
        # x = x.reshape(-1, 20)

        x = self.fc4(x)  # 输出14位二进制数
        # features.append(x.clone().detach())

        return x, features

def test_model():
    # 创建模型实例
    model = ParkingModel()

    # 生成随机输入张量（假设为一批次大小为1的图像数据）
    input_tensor = torch.randn(1, 3, 360, 360)

    # 将模型设置为评估模式
    model.eval()

    # 使用模型进行推理
    with torch.no_grad():
        output, _ = model(input_tensor)

    # 打印输出
    print("模型输出:", output)

# 调用测试函数
if __name__ == '__main__':
    test_model()
