import gymnasium as gym
import gym_donkeycar
import numpy as np
import cv2
import os
import paddle

# 导入自定义模型
from model import DeepDriveModel

# 环境与模拟器配置
exe_path = "../DonkeySimWin/donkey_sim.exe"
port = 9091
conf = {"exe_path": exe_path, "port": port}
env = gym.make("donkey-generated-roads-v0", conf=conf)
# 重置当前场景，获取初始观测值 (obs 即为相机捕捉的图像)
obs, info = env.reset()

# 模型加载与配置
paddle.set_device('cpu')
model = DeepDriveModel()
checkpoint_path = './model/best_model.pdparams'
checkpoint = paddle.load(checkpoint_path)
model.set_state_dict(checkpoint)
model.eval()

# 初始化车辆动作
action = np.array([0.0, 0.2])
# 执行初始动作，获取第一帧有效的图像数据
obs, reward, terminated, truncated, info = env.step(action)

# 自动驾驶循环
for step in range(2000):
    # obs 是 numpy 数组，必须用 .copy() 确保内存连续，否则 paddle.to_tensor 可能报错
    # stop_gradient=True: 推理阶段不需要计算梯度，设为 True 节省内存
    img = paddle.to_tensor(obs.copy(), dtype='float32', stop_gradient=True)
    # 像素值归一化到 [0.0, 1.0]
    img = img / 255.0
    # 维度转换: (H, W, C) -> (C, H, W)，符合模型输入规范
    img = img.transpose([2, 0, 1])
    # 增加 Batch 维度: (C, H, W) -> (1, C, H, W)
    img = img.unsqueeze(0)

    # 模型推理
    with paddle.no_grad():
        # 前向推理，并移除 Batch 维度，转回 numpy 数组
        pre_labels = model(img).squeeze(0).numpy()
    # 提取预测的转向角
    steering_angle = pre_labels[0]
    print("当前转向角度：", steering_angle)

    # 乘以缩放因子 factor，用于调整转向的灵敏度
    factor = 1.5
    # 保持油门 0.2 不变
    action = np.array([steering_angle * factor, 0.2])
    # 将动作输入模拟器，获取下一帧图像及状态
    obs, reward, terminated, truncated, info = env.step(action)

    # 如果发生碰撞或驶出赛道，环境会返回 terminated=True，此时应重置环境
    if terminated or truncated:
        print(f"在第 {step} 步发生碰撞或越界，重置环境...")
        obs, info = env.reset()

# 退出场景
env.close()
