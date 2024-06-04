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

stop_flag = False
stop_count = 0

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
def control_with_neural_network(source: str):
    global stop_flag, stop_count

    frame_count = 0
    while not stop_flag:
        cap = cv2.VideoCapture(source)
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.warpPerspective(frame, M, (360, 360))
        frame_count += 1
        frame_receive_time = time.time()
        motion_index = predict_direction(frame, model, transform, frame_receive_time)
        motion = classDic[motion_index]
        con = Controller(port="COM3", vel=10)
        con.send_command(motion)
        time.sleep(3)

        con.stop_vehicle()

        if motion == 4 or motion == [4]:
            stop_count += 1

        if stop_count >= 5:
            stop_flag = True
            time.sleep(1)  # 停止一秒钟

            if stop_count >= 50:
                break
        cap.release()

    # cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_source ='rtsp://admin:Erleng921010@192.168.139.1/main/Channels/1'
    control_with_neural_network(video_source)
