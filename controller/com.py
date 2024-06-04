import serial
import time

# 打开串口
port = 'COM9'
baudrate = 9600
s = serial.Serial(port, baudrate)

# 发送数据
for i in range(10):
    data = bytearray([0] * 46)  # 生成指令
    data[45] = 128  # 0x80 结束
    for id in range(10):  # 0-128 正，256-128 负，越靠近 128 速度越大
        data[3 * id] = 0  # vy 这里 y 和 x 是反过来的
        data[3 * id + 1] = 20  # vx
        data[3 * id + 2] = 0  # omega
    s.write(data)  # 发送数据
    time.sleep(0.5)  # 暂停 0.5s

# 结束时发零
data = bytearray([0] * 46)
data[45] = 128
s.write(data)

# 关闭串口
s.close()
