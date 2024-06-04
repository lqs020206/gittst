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

# 初始化神经网络模型
model = ParkingModel()
model.load_state_dict(torch.load('best_parking_spot_detection_model4.pth'))

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

def control_with_neural_network(source: str):
    global stop_flag, stop_count
    cap = cv2.VideoCapture(source)
    frame_count = 0
    while not stop_flag:
        ret, frame = cap.read()

        if not ret:
            break
        frame_count += 1

        motion_index = predict_direction(frame, model, transform, frame_receive_time)
        motion = classDic[motion_index[0]]
        con = Controller(port="COM3", vel=10)
        con.send_command(motion)
        time.sleep(0.5)

        con.stop_vehicle()

        if motion == 4 or motion == [4]:
            stop_count += 1

        if stop_count >= 5:
            stop_flag = True
            time.sleep(1)  # 停止一秒钟

            if stop_count >= 50:
                break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_source ='rtsp://admin:Erleng921010@192.168.139.1/main/Channels/1'
    control_with_neural_network(video_source)
