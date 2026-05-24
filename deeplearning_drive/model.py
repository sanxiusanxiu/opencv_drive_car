import paddle
import paddle.nn as nn


class DeepDriveModel(nn.Layer):
    def __init__(self, ):
        super(DeepDriveModel, self).__init__()
        # 输入图像原始尺寸为 (120, 160)，经过卷积下采样后特征图尺寸为 (8, 13)
        self.input_height = 120
        self.input_width = 160
        self.flatten_dim = 64 * 8 * 13
        # 采用5层卷积 + ELU()激活，前3层使用stride=2进行下采样代替池化层
        self.conv_layers = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2D(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2D(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2D(in_channels=48, out_channels=64, kernel_size=3),
            nn.ELU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3),
            # Dropout随机丢弃一部分神经元来减少神经网络复杂度，防止过拟合
            nn.Dropout(p=0.5),
        )
        # 将高维特征逐步映射到 1 维的转向角输出
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=self.flatten_dim, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            # 最后一层不需要激活函数，因为这是回归任务，输出值为任意实数
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        # 调整输出特征的维度，设置为模型预期的输入尺寸 (N, C, H, W)
        # 这一步可以兼容不同形状的输入，并确保经过卷积后特征图尺寸计算正确
        x = paddle.reshape(x, [x.shape[0], 3, self.input_height, self.input_width])
        # 卷积特征提取
        x = self.conv_layers(x)
        # 展平特征图: 将 (N, C, H, W) 展平为 (N, C*H*W)，使用 -1 自动推断展平后的维度长度
        x = paddle.reshape(x, [x.shape[0], -1])
        # 全连接回归
        x = self.linear_layers(x)

        return x
