import cv2
import time
import datetime
from torchvision import transforms
from predict import predict_direction, classDic, classDic_m
from controller.Controller import Controller
from model import ParkingModel
import torch
import math
import numpy as np

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
# 定义变换矩阵
src_pts = np.array([
    [364, 100],
    [12
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 随机颜色生成函数，用于绘制光流的轨迹
def generate_color():
    return np.random.randint(0, 255, 3).tolist()

# 读取第一帧并寻找初始特征点
ret, first_frame = cap.read()
gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(gray_first_frame, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# 创建一个空白图像用于绘制光流轨迹
mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame2model = cv2.warpPerspective(frame, M, (360, 360))
    frame_receive_time=0
    motion_index = predict_direction(frame2model, model, transform, frame_receive_time)
    motion = classDic[motion_index]

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

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_first_frame, gray_frame, p0, None, **lk_params)

    # 选择好的点和新的点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        pt1 = (int(a), int(b))
        pt2 = (int(c), int(d))
        mask = cv2.line(mask, pt1, pt2, generate_color(), 2)
        center = (int(a), int(b))
        frame = cv2.circle(frame, center, 5, generate_color(), -1)



    # 将光流轨迹叠加到原始帧上
    result_frame = cv2.add(frame, mask)
    result_frame = cv2.arrowedLine(result_frame, start_point, end_point, color=(0, 255, 0), thickness=3)
    # 将带有轨迹的帧写入输出视频文件
    out.write(result_frame)

    # cv2.imshow('Optical Flow', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 更新特征点和当前帧
    gray_first_frame = gray_frame.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
out.release()
cv2.destroyAllWindows()
