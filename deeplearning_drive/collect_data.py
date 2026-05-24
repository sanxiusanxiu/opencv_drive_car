import gymnasium as gym
import gym_donkeycar
import numpy as np
import cv2
import os

# 导入自定义库
from img_analysis import region_of_interest, average_lines, compute_steer_angle

# 创建数据集目录
dataset_dir = './dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# 设置模拟器
exe_path = "../DonkeySimWin/donkey_sim.exe"
port = 9091
conf = {"exe_path": exe_path, "port": port}
env = gym.make("donkey-generated-roads-v0", conf=conf)
# 重置当前场景
obs, info = env.reset()

# 开始启动，并获取首帧图像；第1个值是转向值，第2个值是油门值
action = np.array([0, 0.2])
obs, reward, terminated, truncates, info = env.step(action)

# 运行2000次动作
pic_index = len(os.listdir(dataset_dir))
for t in range(2000):
    # 高斯滤波去噪
    frame = cv2.GaussianBlur(obs, (5, 5), 1)

    # 转换图像到HSV空间
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # 特定颜色区域检测
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 边缘检测
    yellow_edge = cv2.Canny(yellow_mask, 200, 400)
    white_edge = cv2.Canny(white_mask, 200, 400)

    # 感兴趣区域提取
    yellow_cropped = region_of_interest(yellow_mask, side="left")
    white_cropped = region_of_interest(white_mask, side="right")

    # 直线检测
    rho = 1
    angle = np.pi / 180
    min_thr = 10
    yellow_lines = cv2.HoughLinesP(yellow_cropped, rho, angle, min_thr, np.array([]), minLineLength=8, maxLineGap=8)
    white_lines = cv2.HoughLinesP(white_cropped, rho, angle, min_thr, np.array([]), minLineLength=8, maxLineGap=8)

    # 小线段聚类
    yellow_lane = average_lines(frame, yellow_lines, direction="left")
    white_lane = average_lines(frame, white_lines, direction="right")

    # 计算转向角
    steering_angle = compute_steer_angle(yellow_lane, white_lane, height, width)
    print("当前转向角度：", steering_angle)
    # 油门值恒定
    action = np.array([steering_angle, 0.2])

    # 记录图像和转向角
    img_path = dataset_dir + "/{:d}_{:.4f}.jpg".format(pic_index, steering_angle)
    # 注意gym-donkeycar官网给出的例子里obs就是图片，业内常使用frame作为摄像头捕捉的图片
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, obs)
    pic_index += 1
    obs, reward, terminated, truncates, info = env.step(action)

# 运行结束后重置当前场景
obv = env.reset()
