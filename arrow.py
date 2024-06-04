import time
import threading
import cv2
import datetime
from torchvision import transforms
from predict import predict_direction, classDic, classDic_m
from controller.Controller import Controller
from model import ParkingModel
import torch
import math
import numpy as np
# 初始化神经网络模型

model = ParkingModel()
model.load_state_dict(torch.load('best_parking_spot_detection_model6.pth'))
motionDir = {
    0: [-1, 1],
    1: [0, math.sqrt(2)],
    2: [1, 1],
    3: [-math.sqrt(2), 0],
    4: [0, 0],
    5: [math.sqrt(2), 0],
    6: [-1, -1],
    7: [0, -math.sqrt(2)],
    8: [1, -1],
}
transform = transforms.Compose([
    transforms.Resize([360, 360]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
import random
# 定义变换矩阵
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
cap = cv2.VideoCapture("D:/OneDrive - sjtu.edu.cn/桌面/视频0225/0000slow.mp4")
output_file = "D:/OneDrive - sjtu.edu.cn/桌面/视频0225/0000slow1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, 20.0, (1920, 1080))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame2model = cv2.warpPerspective(frame, M, (360, 360))
    frame_receive_time=0
    motion_index = predict_direction(frame2model, model, transform, frame_receive_time)
    motion = classDic[motion_index[0]]

    # 设置箭头的起点和终点坐标
    width = 1920
    height = 1080
    start_point = (width // 2, height // 2)
    arrow_length = 100

    arrow_direction = motionDir[motion]
    end_point = (
        int(start_point[0] + arrow_length * arrow_direction[0]),
        int(start_point[1] - arrow_length * arrow_direction[1])
    )

    # 在图像上绘制箭头
    frame_with_arrow = cv2.arrowedLine(frame, start_point, end_point, color=(0, 255, 0), thickness=3)
    out.write(frame_with_arrow)
