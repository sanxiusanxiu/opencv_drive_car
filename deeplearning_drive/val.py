import os
import paddle
import time
from paddle.vision import transforms
from data_loader import DeepDriveDataset
from model import DeepDriveModel

paddle.set_device('cpu')
data_transform = transforms.Compose([
    transforms.ToTensor(),
])
# 验证数据集
val_dataset = DeepDriveDataset(mode='val', transform=data_transform)
# 验证数据加载器
val_loader = paddle.io.DataLoader(
    dataset=val_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False, return_list=True
)

# 加载模型
model = DeepDriveModel()
# checkpoint_path = './model/model.pdparams'
checkpoint_path = './model/best_model.pdparams'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"未找到模型权重文件: {checkpoint_path}")
checkpoint = paddle.load(checkpoint_path)
model.set_state_dict(checkpoint)

# 模型验证
model.eval()
loss_fn = paddle.nn.MSELoss()
total_mse = 0.0
total_samples = 0

start_time = time.time()
# 禁用梯度计算，减少内存占用并加速推理
with paddle.no_grad():
    for batch_id, data in enumerate(val_loader):
        # return_list=True，通过下标取值更安全
        imgs = data[0]
        labels = data[1]

        # 前向推理
        pre_labels = model(imgs)
        # 计算 MSE 损失
        loss = loss_fn(pre_labels, labels)
        # 乘以当前batch的样本数，还原为该batch的总损失
        total_mse += loss.item() * imgs.shape[0]
        total_samples += imgs.shape[0]

end_time = time.time()
total_time = end_time - start_time
# 计算整个验证集的平均 MSE (总损失 / 总样本数)
avg_mse = total_mse / total_samples
# 计算平均单张图片用时
avg_time_per_sample = total_time / total_samples
print(f"验证集总样本数: {total_samples}")
print(f"验证集平均 MSE: {avg_mse:.6f}")
print(f"平均单张图片用时: {avg_time_per_sample:.4f} 秒")
