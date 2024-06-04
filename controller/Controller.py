import serial
import time
from numpy import ndarray
import numpy as np
from datetime import datetime

motionDir = {
    0: [-1,  1],
    1: [ 0,  1],
    2: [ 1,  1],
    3: [-1,  0],
    4: [ 0,  0],
    5: [ 1,  0],
    6: [-1, -1],
    7: [ 0, -1],
    8: [ 1, -1],
}
"""
分类结果与运动方向的映射，x为向右，y为向左

格式 index: [x, y]
"""


class Controller(object):
    def __init__(self, port="COM3", baudrate=9600, vel=20):
        """
        初始化小车控制器
        :param port: 串口号
        :param baudrate: 波特率
        :param vel: 小车默认速度
        """
        try:
            self.ser = serial.Serial(port, baudrate)
            self.vel = vel
            print("Serial connection established on port", port)
        except serial.SerialException:
            self.ser = fakeSerial(port, baudrate)
            self.vel = vel
            # print("Failed to connect to serial port", port)

    def send_command(self, index, t=None):
        """
        根据索引向串口发送控制命令
        :param index: 运动模式索引
        :return:
        """

        print("class={}".format(index))
        if type(index) is not int:
            index = index[0]

        # 未识别的类别：停止行动
        if index not in motionDir:
            self.stop_vehicle()

        # 构造控制数据包
        data = bytearray([0] * 46)
        data[45] = 128              # 0x80 结束
        for i in range(12):                                             # 0-128 正，256-128 负，越靠近 128 速度越大
            data[3 * i] = self.vel * motionDir[index][1] % 256          # vy
            data[3 * i + 1] = self.vel * motionDir[index][0] % 256      # vx
            data[3 * i + 2] = 0                                         # omega

        self.ser.write(data)
        # print("1",data)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

        if t:
            time.sleep(t)
            self.stop_vehicle()


    def stop_vehicle(self):
        """
        向小车发送停止数据包
        """
        data = bytearray([0] * 46)
        data[45] = 128
        self.ser.write(data)
        time.sleep(0.1)

    def close(self):
        self.ser.close()


class fakeSerial(object):
    def __init__(self, port, baudrate):
        """
        用于在Serial创建失败时替代Serial的假串口
        """
        pass

    def write(self, data):
        decimal_repr = [byte for byte in data]
        # print("2", decimal_repr)
        pass

    def close(self):
        pass
