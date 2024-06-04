import datetime
import os
import time
import numpy as np
import multiprocessing
from multiprocessing import Process
import cv2
import torch
from torchvision import transforms
from model import ParkingModel
from predict import predict_direction, classDic
from controller.Controller import Controller

model = ParkingModel()
model.load_state_dict(torch.load('model_epoch5p0.05 bs100.pth'))

transform = transforms.Compose([
    transforms.Resize([360, 360]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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

def reading_thread(que, video_source, max_len=50):
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read camera.')
            break
        length = que.qsize()
        if length > max_len:
            t11 = time.time()
            for i in range(length - 8):
                frame_rm = que.get()
            t22 = time.time()
            print('clear img que in', t22-t11, 'seconds.')

        que.put(frame)

def sending_thread(que):
    con = Controller(port="COM3", vel=10)
    stop_count = 0
    last_motion = 0
    stop_flag = False
    while not stop_flag:
        if que.qsize() > 0:
            frame = que.get()
            frame = cv2.warpPerspective(frame, M, (360, 360))
            motion_index = predict_direction(frame, model, transform, time.time())
            motion = classDic[motion_index]
            con.send_command(motion)

            time.sleep(0.1)

            if motion == 4 and last_motion == 4:
                stop_count += 1
            if stop_count > 3:
                stop_flag = True
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
            last_motion = motion
        else:
            print('wait for frame in sending thread')
            con.stop_vehicle()
    print('stop')

if __name__ == "__main__":

    video_source = "rtsp://admin:Erleng921010@192.168.139.1/main/Channels/1"
    
    manager = multiprocessing.Manager()
    que = manager.Queue()

    t1 = Process(target=reading_thread, args=(que, video_source, 50))
    t2 = Process(target=sending_thread, args=(que,))

    t1.start()
    t2.start()

    t2.join()
    t1.terminate()
