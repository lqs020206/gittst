

import os
import cv2
import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import ParkingModel
import time
classDic = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
classDic_m = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}
positions = {
        0: 0,
        1000: 1,
        1100: 2,
        1110: 3,
    }
positions1 = [
        "0000", "1000", "1100", "1110"
    ]


# def predict_direction(frame: np.ndarray, model: torch.nn.Module, transform: transforms.Compose) -> int or list:
#     im = Image.fromarray(frame)  # 假设frame是一个图像帧的numpy数组
#     im = transform(im)  # [C, H, W]
#     im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
#
#     with torch.no_grad():
#         outputs = model(im)
#         predict = torch.max(outputs, dim=1)[1].numpy()
#
#     return predict
def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def predict_direction(frame: np.ndarray, model: torch.nn.Module, transform: transforms.Compose,frame_receive_time: float) -> int or list:
    t1 = time.time()
    im = Image.fromarray(frame)  # 假设frame是一个图像帧的numpy数组
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]


    # im = im.to("cuda")  # 将输入数据移动到GPU上
    # model.to("cuda")  # 将模型移动到GPU上

    with torch.no_grad():
        model.apply(apply_dropout)
        output, _ =model(im)


        outputs = output[:, -9:]
        outputs1 = output[:, :4]
        predict = torch.max(outputs, dim=1)[1].item()  # 将预测结果移动到CPU上
        park=positions1[torch.max(outputs1, dim=1)[1].item()]



    predict_time = time.time()
    receive_delay = predict_time - frame_receive_time
    # print("receive_delay",receive_delay)
    # threshold = 0.5  # 设置阈值
    #
    # first_four_outputs = (output[:, :4] >= threshold).int()
    # #print(outputs)
    # first_four_outputs = first_four_outputs.reshape(-1)
    # pre2=time.time()
    # print("sec2", pre2-predict_time)
    # binary_test_first_four_outputs = binary_test_first_four_outputs.reshape(-1)
    print('车位',park)
    print('方向',predict)
    print("Receive delay:", receive_delay)
    # t2 = time.time()
    # print("t3",t2-t1)
    return predict

if __name__ == '__main__':
    folder_path = "E:/Downloads/192.168.139.1_01_20231206161400535.jpeg"
    net, transform = initialize_model()

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        index = predict_direction(img, net, transform)
        print("class {}".format(classDic[index[0]]))
