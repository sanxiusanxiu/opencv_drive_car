import numpy as np
import cv2
import paddle
from typing import List, Tuple, Optional, Callable


class DeepDriveDataset(paddle.io.Dataset):
    """
    自动驾驶端到端转向角预测数据集
    继承自 paddle.io.Dataset，用于读取图像及对应的转向角标签
    """

    def __init__(self, mode: str, transform: Optional[Callable] = None):
        super(DeepDriveDataset, self).__init__()
        self.mode = mode.lower()
        self.transform = transform

        # 确保模式合法
        assert self.mode in {"train", "val"}, f"模式必须是 'train' 或 'val', 但收到了 {self.mode}"

        # 根据模式选择对应的列表文件
        if self.mode == "train":
            file_path = "./train.txt"
        else:
            file_path = "./val.txt"

        # 存储数据信息：每个元素为 [图像路径, 转向角标签]
        self.data_list: List[Tuple[str, float]] = []

        # 读取数据列表文件
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 按空格分割，假设格式为: 图像路径 转向角
                parts = line.split(" ")
                img_path = parts[0]
                # 使用 float() 转换标签，并兼容可能存在的多空格情况
                label = float(parts[-1])
                self.data_list.append((img_path, label))

    def __getitem__(self, index: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
        # 获取当前索引的图像路径和标签
        img_path, label_val = self.data_list[index]

        # 读取图像 (OpenCV 默认读取为 BGR 格式)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"图像读取失败，请检查路径: {img_path}")
        # 转换为 RGB 格式 (深度学习模型通常默认使用 RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 图像预处理
        if self.transform is not None:
            img = self.transform(img)

        # 使用 [label_val] 构建形状为 (1,) 的一维张量，方便后续计算损失
        label = paddle.to_tensor([label_val], dtype="float32")

        return img, label

    def __len__(self) -> int:
        return len(self.data_list)
