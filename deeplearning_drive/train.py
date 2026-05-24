import os
import paddle
from paddle.vision import transforms
from visualdl import LogWriter

from deeplearning_drive.data_loader import DeepDriveDataset
from deeplearning_drive.model import DeepDriveModel


# 参数配置
paddle.set_device('cpu')
# CPU下适当增大batch_size可提升利用率，太小会导致训练极慢
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

# 准备数据
# 将PIL图像或numpy数组转换为Tensor，并归一化到[0.0, 1.0]
data_transform = transforms.Compose([
    transforms.ToTensor(),
])
# 实例化训练数据集
train_dataset = DeepDriveDataset(mode='train', transform=data_transform)
# 实例化数据加载器
train_loader = paddle.io.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # CPU下建议设为0，避免多进程开销和报错
    num_workers=0,
    # 丢弃最后一个不完整的batch，确保batch_size一致
    drop_last=True,
    # 以list形式返回数据
    return_list=True
)

# 模型、损失函数与优化器
model = DeepDriveModel()
# 定义均方误差损失函数 (回归任务常用)
loss_fn = paddle.nn.MSELoss()
# 定义Adam优化器
optimizer = paddle.optimizer.Adam(learning_rate=LEARNING_RATE, parameters=model.parameters())
# 创建VisualDL日志记录器
writer = LogWriter(logdir='./log')

# 保存模型
# 确保保存模型的目录存在
model_dir = './model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
best_model_path = os.path.join(model_dir, 'best_model.pdparams')
last_model_path = os.path.join(model_dir, 'last_model.pdparams')
# 初始化最小损失值为无穷大（这样第一轮的损失值一定比它小，就能触发保存）
best_loss = float('inf')

# 开始训练
print("开始训练...")
for epoch in range(EPOCHS):
    # 设置模型为训练模式（启用Dropout、BatchNorm等）
    model.train()
    # 统计单轮总损失值
    loss_epoch = 0.0
    total_batches = 0
    for batch_id, (imgs, labels) in enumerate(train_loader):
        # 前向传播，计算预测值
        pre_labels = model(imgs)
        # 计算损失值 (注意：这里使用loss_fn而不是loss，避免与变量名冲突)
        loss_val = loss_fn(pre_labels, labels)
        # 反向传播
        optimizer.clear_grad()
        loss_val.backward()
        # 更新参数
        optimizer.step()
        # 累加当前batch的损失
        loss_epoch += loss_val.item()
        total_batches += 1

    # 手动释放不需要的中间变量，缓解CPU内存压力
    del imgs, labels, pre_labels, loss_val
    # 计算每个Batch的平均损失 (比除以数据集样本数更直观反映训练状态)
    loss_epoch_avg = loss_epoch / total_batches
    # 记录日志到VisualDL
    writer.add_scalar(tag='MSELoss', value=loss_epoch_avg, step=epoch)

    # 保存最佳模型
    if loss_epoch_avg < best_loss:
        # 如果当前轮次的损失小于历史最小损失，则更新最小损失并保存模型
        best_loss = loss_epoch_avg
        paddle.save(model.state_dict(), best_model_path)
        print(
            f"轮次: {epoch + 1}/{EPOCHS}, 损失值: {loss_epoch_avg:.6f} 发现更优模型，已保存！(当前最佳: {best_loss:.6f})")
    else:
        # 否则只打印普通训练信息
        print(f"轮次: {epoch + 1}/{EPOCHS}, 损失值: {loss_epoch_avg:.6f} (当前最佳: {best_loss:.6f})")

# 保存最后一轮的模型，方便断点续训或对比
paddle.save(model.state_dict(), last_model_path)
print(f"训练结束！最后一轮模型参数已保存至: {last_model_path}")
print(f"最佳模型参数已保存至: {best_model_path}，最佳损失值: {best_loss:.6f}")

# 关闭日志记录器
writer.close()
