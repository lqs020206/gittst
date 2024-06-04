import os
import cv2
import numpy as np

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

# 指定原始图片文件夹和目标文件夹路径
source_folder = "F:/Code/tingchechang/datayoloclassify"
target_folder = "F:/Code/tingchechang/dataclassify"

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历第一级子文件夹
for folder_name_lvl1 in os.listdir(source_folder):
    folder_path_lvl1 = os.path.join(source_folder, folder_name_lvl1)
    if os.path.isdir(folder_path_lvl1):
        # 构造目标第一级文件夹路径
        target_folder_path_lvl1 = os.path.join(target_folder, folder_name_lvl1)
        if not os.path.exists(target_folder_path_lvl1):
            os.makedirs(target_folder_path_lvl1)

        # 遍历第二级子文件夹
        for folder_name_lvl2 in os.listdir(folder_path_lvl1):
            folder_path_lvl2 = os.path.join(folder_path_lvl1, folder_name_lvl2)
            if os.path.isdir(folder_path_lvl2):
                # 构造目标第二级文件夹路径
                target_folder_path_lvl2 = os.path.join(target_folder_path_lvl1, folder_name_lvl2)
                if not os.path.exists(target_folder_path_lvl2):
                    os.makedirs(target_folder_path_lvl2)

                # 遍历每张图片
                for filename in os.listdir(folder_path_lvl2):
                    if filename.endswith(".jpg"):
                        # 读取图片
                        img_path = os.path.join(folder_path_lvl2, filename)
                        img = cv2.imread(img_path)
                        # 进行透视变换
                        dst = cv2.warpPerspective(img, M, (360, 360))
                        # 构造目标图片路径
                        target_img_path = os.path.join(target_folder_path_lvl2, filename)
                        # 保存处理后的图片
                        cv2.imwrite(target_img_path, dst)

print("预处理完成")
