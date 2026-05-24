import gymnasium as gym
import gym_donkeycar
import numpy as np
import cv2
import os

exe_path = "../DonkeySimWin/donkey_sim.exe"
port = 9091
conf = {"exe_path": exe_path, "port": port}

# 设置模拟器环境，选择赛道
env = gym.make("donkey-generated-roads-v0", conf=conf)

# 重置当前场景
obs, info = env.reset()
for t in range(100):
    # 定义控制动作，0.3表示转向，0.5表示油门
    action = np.array([0.3, 0.5])
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    # 取一张图像保存
    if t == 7:
        img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output/test_drive.jpg', img)

# 退出场景
env.close()
