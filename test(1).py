import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from navi_eval import navi_eval
import math
import queue

def h(curr_pos, goal_pos, method='manhattan'):
    if method == 'euclidean':
        return np.sqrt((goal_pos[0]-curr_pos[0])**2 + (goal_pos[1]-curr_pos[1])**2)
    elif method == 'manhattan':
        return np.abs(goal_pos[0]-curr_pos[0]) + np.abs(goal_pos[1]-curr_pos[1])

def is_valid_pos(pos, world_map):
    w, h = world_map.shape
    in_range = 0<=pos[0]<w and 0<=pos[1]<h
    is_obstacle = world_map[pos[0], pos[1]] == 1
    return in_range and ~is_obstacle

def dilate(world_map, neighbors, expand_size=1):
    if expand_size == 0:
        return world_map
    dilated_map = np.zeros([world_map.shape[0]+2*expand_size,
                            world_map.shape[1]+2*expand_size],
                            dtype=world_map.dtype)
    for i in range(world_map.shape[0]):
        for j in range(world_map.shape[1]):
            if world_map[i, j] == 1:
                dilated_map[i+expand_size, j+expand_size] = 1
                for neighbor in neighbors:
                    for expand_i in range(expand_size+1):
                        dilated_x = i+expand_size+expand_i*neighbor[0]
                        dilated_y = j+expand_size+expand_i*neighbor[1]
                        dilated_map[dilated_x, dilated_y] = 1
    return dilated_map[expand_size:-expand_size, expand_size:-expand_size]

def steer_cost(curr_pos, next_pos, steer_cost_map):
    if steer_cost_map[curr_pos[0], curr_pos[1]] == -2:
        # start_pos case
        return 0
    elif steer_cost_map[next_pos[0], next_pos[1]] == steer_cost_map[curr_pos[0], curr_pos[1]]:
        return -1
    else:
        return 0

def smooth(path, step):
    smooth_path = []
    T = np.linspace(0, 1, step)
    n = len(path) - 1
    for t in T:
        x = 0
        y = 0
        for i in range(n+1):
            x += math.comb(n, i)*math.pow(t, i)*math.pow(1-t, n-i) * path[i][0]
            y += math.comb(n, i)*math.pow(t, i)*math.pow(1-t, n-i) * path[i][1]
        smooth_path.append([x, y])
    return smooth_path

def self_driving_path_planner(world_map, start_pos, goal_pos, expand_size=1):

    open_list = queue.PriorityQueue()
    g = 10000 * np.ones([world_map.shape[0], world_map.shape[1]])
    steer_cost_map = -1 * np.ones([world_map.shape[0], world_map.shape[1]])
    father = -1 * np.ones([world_map.shape[0], world_map.shape[1], 2])
    neighbor8 = np.array([
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1]
    ])
    open_list.put([h(start_pos, goal_pos), start_pos])
    g[start_pos[0], start_pos[1]] = 0
    steer_cost_map[start_pos[0], start_pos[1]] = -2

    dilated_map = dilate(world_map, neighbor8, expand_size=expand_size)

    while not open_list.empty():
        item = open_list.get()
        curr_pos = item[1]
        if curr_pos == goal_pos:
            break
        for steer_idx, neighbor in enumerate(neighbor8):
            next_pos = curr_pos + neighbor
            next_pos = next_pos.tolist()
            if not is_valid_pos(next_pos, dilated_map):
                continue
            delta_d = np.sqrt(neighbor[0]**2+neighbor[1]**2)
            if g[next_pos[0], next_pos[1]] > g[curr_pos[0], curr_pos[1]] + delta_d:
                steer_cost_map[next_pos[0], next_pos[1]] = steer_idx
                g[next_pos[0], next_pos[1]] = g[curr_pos[0], curr_pos[1]] + delta_d
                new_f = h(next_pos, goal_pos) + \
                    g[next_pos[0], next_pos[1]] + \
                    steer_cost(curr_pos, next_pos, steer_cost_map)
                open_list.put([new_f, next_pos])
                father[next_pos[0], next_pos[1]] = curr_pos
    
    path = []
    pos = goal_pos
    while father[pos[0], pos[1], 0] != -1:
        path.append(pos)
        pos = father[pos[0], pos[1]].astype(np.int32).tolist()
    path = path[::-1]

    # path = smooth(path, step=100)
    return path, dilated_map

def vis_path(map, path):
    # Visualize the map and path.
    obstacles_x, obstacles_y = [], []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(start_pos[0], start_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def create_map(field_size, obstacle):
    map = np.zeros(field_size, dtype=np.int32)
    map[0, :] = 1
    map[-1, :] = 1
    map[:, 0] = 1
    map[:, -1] = 1

    for obj in obstacle:
        for x in range(obj[0], obj[2]+1):
            for y in range(obj[1], obj[3]+1):
                map[x, y] = 1

    return map

def add_obstacle(all_list, all_center, i, r):
    all_list.append([
        round(all_center[i][0]-r),
        round(all_center[i][1]-r),
        round(all_center[i][0]+r),
        round(all_center[i][1]+r),
    ])

def esacpe_from_obstacle(pos, obstacle_center):
    distance = [[i, h(pos, obstacle_center[i])] for i in range(len(obstacle_center))]
    sorted_distance = sorted(distance, key=lambda x:x[1])
    nearest_obstacle = obstacle_center[sorted_distance[0][0]]
    return 8-navi_eval([pos, nearest_obstacle])

def parse_timestamp(timestamp_str):
    if timestamp_str == 'Time':
        return None  # If the timestamp string is 'Time', return None

    formats_to_try = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]

    for format_str in formats_to_try:
        try:
            return datetime.strptime(timestamp_str, format_str)
        except ValueError:
            pass  # Ignore the error and continue to the next format

    return None  # If both formats fail, return None


def match_images_with_positions(position_data, images):
    image_dict = {}  # 存储图像名称和位置信息的映射

    # 保存当前已找到图像的索引
    current_image_index = 0

    # 计数器，用于记录当前处理的图像数量
    processed_images_count = 0

    for image_name in tqdm(images):
        if image_name is None:
            continue  # 如果image_name为None，则跳过此迭代

        try:
            image_timestamp = parse_image_timestamp(image_name)
            # print(image_timestamp)

            # 从当前图像的索引开始寻找下一个图像的位置
            for index in range(current_image_index, len(position_data)):
                try:
                    row = position_data.iloc[index]
                    timestamp_str = str(row[0])
                    # 检查row[10]和row[11]是否为数字
                    if not isinstance(row[10], (int, float)) or not isinstance(row[11], (int, float)):
                        continue

                    timestamp, position = parse_timestamp(timestamp_str), (float(row[10]) / 10, float(row[11]) / 10)
                    if timestamp is None:
                        continue  # 如果时间戳解析失败，则跳过此迭代
                    if check_timestamp_match(timestamp, image_timestamp):
                        image_dict[image_name] = position
                        current_image_index = index + 1  # 更新当前图像的索引，从下一个位置开始查找
                        processed_images_count += 1  # 增加已处理图像的计数
                        # print(processed_images_count)
                        break  # 找到相应的图像后退出循环
                except (ValueError, TypeError):
                    # 处理与时间戳或位置解析相关的异常
                    continue
        except Exception as e:
            # 处理与parse_image_timestamp函数相关的异常，请用实际的异常类型替换SomeSpecificException
            continue

    return image_dict

def parse_image_timestamp(image_name):
    try:
        time_obj = datetime.strptime(image_name, "%Y-%m-%d_%H-%M-%S-%f.jpg")
        return time_obj
    except ValueError as e:
        # print(f"Error parsing image timestamp {image_name}: {e}")
        return None

def check_timestamp_match(ts1, ts2):
    return abs((ts1 - ts2).total_seconds()) <= 0.01


if __name__ == '__main__':
    ###############################
    all_park_center = [
        [30, 60],
        [30, 7],
        [69, 7],
        [66, 56],
        [27, 32]
    ]  # left_top, left_bottom, right_bottom, right_top, middle
    obstacle_r = 3
    car_r = 3

    # setting = '0000'
    ##############################
    for setting in ['0000', '1000', '1100', '1110']:
        obstacle = []
        add_obstacle(obstacle, all_park_center, 4, obstacle_r)
        obstacle_center = [all_park_center[4]]
        if setting == '0000':
            goal_pos = [all_park_center[0][0], all_park_center[0][1]-3]
            image_folder=("F:/Code/cc neural network/zhenbaocun")
            position_file_path=("D:/OneDrive - sjtu.edu.cn/桌面/位置/1.xlsx")
            output_folder=("F:/Code/tingchechang/datanew/0000")
        elif setting == '1000':
            add_obstacle(obstacle, all_park_center, 0, obstacle_r)
            obstacle_center.append(all_park_center[0])
            goal_pos = [all_park_center[1][0], all_park_center[1][1]+3]
            image_folder=("F:/Code/cc neural network/zhenbaocun1")
            position_file_path=("D:/OneDrive - sjtu.edu.cn/桌面/位置/2.xlsx")
            output_folder=("F:/Code/tingchechang/datanew/1000")
        elif setting == '1100':
            add_obstacle(obstacle, all_park_center, 0, obstacle_r)
            add_obstacle(obstacle, all_park_center, 1, obstacle_r)
            obstacle_center.append(all_park_center[0])
            obstacle_center.append(all_park_center[1])
            goal_pos = [all_park_center[2][0], all_park_center[2][1]+3]
            image_folder=("F:/Code/cc neural network/zhenbaocun2")
            position_file_path=("D:/OneDrive - sjtu.edu.cn/桌面/位置/3.xlsx")
            output_folder=("F:/Code/tingchechang/datanew/1100")
        elif setting == '1110':
            add_obstacle(obstacle, all_park_center, 0, obstacle_r)
            add_obstacle(obstacle, all_park_center, 1, obstacle_r)
            add_obstacle(obstacle, all_park_center, 2, obstacle_r)
            obstacle_center.append(all_park_center[0])
            obstacle_center.append(all_park_center[1])
            obstacle_center.append(all_park_center[2])
            goal_pos = [all_park_center[3][0], all_park_center[3][1]-3]
            image_folder=("F:/Code/cc neural network/zhenbaocun3")
            position_file_path=("D:/OneDrive - sjtu.edu.cn/桌面/位置/4.xlsx")
            output_folder=("F:/Code/tingchechang/datanew/1110")
        map = create_map((100, 70), obstacle=obstacle)
        goal_pos = [round(goal_pos[0]), round(goal_pos[1])]

        images = os.listdir(image_folder)
        position_data = pd.read_excel(position_file_path, header=None)
        image_dict = match_images_with_positions(position_data,images)

        for image_name in tqdm(image_dict, desc=f'Processing images {setting}'):
            start_pos = image_dict[image_name]
            if start_pos[0] == 0 or start_pos[1] == 0:
                continue
            start_pos = [start_pos[0]/5, start_pos[1]/5]

            if h(start_pos, goal_pos, method='euclidean') < 3:
                # 到达
                navi_dir = 4
            else:
                start_pos = [round(start_pos[0]), round(start_pos[1])]
                path, dilated_map = self_driving_path_planner(map, start_pos, goal_pos, car_r)
                # 在膨胀地图的障碍物中
                if not is_valid_pos(start_pos, dilated_map):
                    navi_dir = esacpe_from_obstacle(start_pos, obstacle_center)
                else:
                    navi_dir = navi_eval(path)

            target_folder = os.path.join(output_folder, str(navi_dir))
            os.makedirs(target_folder, exist_ok=True)
            image_path = os.path.join(image_folder, image_name)
            output_path = os.path.join(target_folder, image_name)
            shutil.copy(image_path, output_path)