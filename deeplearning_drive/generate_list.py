import random
from pathlib import Path

# 配置参数
IMG_FOLDER = Path("./dataset")
TRAIN_RATIO = 0.8
RANDOM_SEED = 256


def iter_files(directory, ext=None):
    """
    递归遍历目录，生成满足扩展名条件的文件路径

    :param directory: 根目录路径 (Path 对象或字符串)
    :param ext: 指定后缀名，如 '.jpg'。若为 None 则返回所有文件
    :return: 生成器，产出文件的 Path 对象
    """
    root = Path(directory)
    for path in root.rglob('*'):
        if path.is_file():
            if ext is None or path.suffix.lower() == ext.lower():
                yield path


def create_data_list(file_list, mode, output_dir="."):
    """
    将文件列表写入 txt 文件，每行格式为：图片路径 角度值

    :param file_list: 包含 (图片路径, 角度) 元组的可迭代对象
    :param mode: 文件名标识（如 'train' 或 'val'）
    :param output_dir: 输出目录，默认为当前目录
    """
    output_path = Path(output_dir) / f"{mode}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for img_path, angle in file_list:
            f.write(f"{img_path} {angle}\n")

    print(f"{output_path} 已生成")


# 检索文件
jpg_generator = iter_files(IMG_FOLDER, ".jpg")
jpg_list = list(jpg_generator)
print(f"检索到 {len(jpg_list)} 个 jpg 文件\n")

# 解析转向值
file_list = []
for jpg_path in jpg_list:
    # stem 获取无后缀文件名
    angle = jpg_path.stem.split("_")[-1]
    # 将路径统一为正斜杠字符串
    # as_posix() 等同于 replace("\\", "/")，但更官方、更安全
    img_path_str = jpg_path.as_posix()
    file_list.append((img_path_str, angle))

# 切分数据
random.seed(RANDOM_SEED)
random.shuffle(file_list)
train_num = int(len(file_list) * TRAIN_RATIO)

train_list = file_list[:train_num]
val_list = file_list[train_num:]

# 创建列表文件
create_data_list(train_list, mode="train")
create_data_list(val_list, mode="val")
