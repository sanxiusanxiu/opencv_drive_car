import cv2
import numpy as np
import math


# 感兴趣区域提取函数
def region_of_interest(img, side="left"):
    """
    提取图像的感兴趣区域（ROI），只保留画面下半部分的指定侧，遮挡其他区域
    :param img: 输入的图像（通常是单通道的边缘检测图或掩膜图）
    :param side: 指定提取哪一侧。"left"提取左下区(找黄线), "right"提取右下区(找白线)
    :return: 只包含感兴趣区域内容的图像
    """
    height, width = img.shape
    # 创建一个与原图大小相同的全黑掩膜（单通道，像素值全为0）
    mask = np.zeros_like(img)
    # 计算图像中心和底部的关键坐标（使用 // 整除，避免浮点数）
    mid_x = width // 2
    mid_y = height // 2

    # ---------------- 构建多边形顶点坐标 ----------------
    # 原点(0,0)在图像左上角：向右为X正方向，向下为Y正方向
    # 提取逻辑：只看画面的下半部分（Y从 mid_y 到 height）
    if side == "left":
        # 左侧区域：X 范围从 0 到 mid_x
        x_start, x_end = 0, mid_x
    else:
        # 右侧区域：X 范围从 mid_x 到 width
        x_start, x_end = mid_x, width

    # 定义多边形的四个顶点（按照顺时针或逆时针顺序连接）
    # 左上角 -> 右上角 -> 右下角 -> 左下角
    polygon = np.array([[
        (x_start, mid_y),
        (x_end, mid_y),
        (x_end, height),
        (x_start, height)
    ]], np.int32)

    # 在全黑的mask上，用白色(255)填充上面定义的多边形区域，形成透视窗
    cv2.fillPoly(mask, polygon, 255)
    # 提取感兴趣区域，cv2.bitwise_and 进行按位与运算：
    # mask中黑色的区域(0)与原图做 AND，会把原图对应区域变成黑色(0)，相当于"遮挡"
    # mask中白色的区域(255)与原图做 AND，会保留原图原本的像素值，相当于"透视"
    cropped_img = cv2.bitwise_and(img, mask)

    return cropped_img


# 由于检测到多条线段，我们只需要两条车道线，可以对检测到的小线段进行聚类和平均
def make_points(img, line_params):
    """
    根据直线的斜率和截距，结合图像高度，计算直线在画面中的起点和终点像素坐标
    :param img: 输入图像（用于获取图像高度）
    :param line_params: 包含斜率和截距的元组/数组 (slope, intercept)
    :return: 包含两个端点坐标的数组 [x1, y1, x2, y2]
    """
    height, width, _ = img.shape
    slope, intercept = line_params

    # 设定直线绘制的纵向范围：从图像底部 (y1) 到图像中线 (y2)
    y1 = height
    y2 = int(height * 0.5)

    # 根据直线方程 y = slope * x + intercept 推导 x = (y - intercept) / slope
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_lines(img, lines, direction):
    """
    对检测到的多条小线段进行聚类和平均，拟合出一条完整的车道线
    :param img: 输入图像
    :param lines: 霍夫变换检测到的线段列表
    :param direction: 车道方向，"left"（左侧黄线）或 "right"（右侧白线）
    :return: 拟合后的车道线端点数组 [x1, y1, x2, y2]，若未检测到则返回空列表
    """
    lane_line_coords = []  # coords，坐标

    # 如果没有检测到任何线段，直接返回空列表
    if lines is None:
        print(f"[{direction}] 没有检测到线段")
        return lane_line_coords

    valid_fits = []

    # 遍历所有检测到的线段
    for line in lines:
        for x1, y1, x2, y2 in line:
            # 只要两个点不在同一垂直线上（防止除以0），就可以拟合直线
            if x1 != x2:
                # np.polyfit 拟合（最小二乘法）一次多项式（直线），返回 [斜率(slope), 截距(intercept)]
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]

                # 画面左侧的线（向右上方倾斜），斜率是负数
                # 画面右侧的线（向左上方倾斜），斜率是正数
                if direction == "left" and slope < 0:
                    valid_fits.append((slope, intercept))
                elif direction == "right" and slope > 0:
                    valid_fits.append((slope, intercept))

    # 遍历完所有线段后，进行平均计算
    if len(valid_fits) > 0:
        # 对所有有效线段的斜率和截距求平均值，axis=0 表示按列求平均
        avg_fit = np.average(valid_fits, axis=0)
        # 将平均后的斜率和截距转换为图像上的像素坐标点
        lane_line_coords = make_points(img, avg_fit)

    return lane_line_coords


# 计算转向角度
def compute_steer_angle(yellow_lane, white_lane, height, width):
    """
    根据检测到的车道线坐标，计算小车的转向角度
    :param yellow_lane: 左侧黄线端点 [x1, y1, x2, y2]，若无则为空数组
    :param white_lane: 右侧白线端点 [x1, y1, x2, y2]，若无则为空数组
    :param height: 图像高度
    :param width: 图像宽度
    :return: 归一化后的转向角度 (范围大致在 -1.0 到 1.0 之间)，若无车道线返回 None
    """
    # 默认无横向偏移（即目标在正前方）
    x_offset = 0
    # 纵向偏移量：取图像下半部分的高度（视线焦点到车底的垂直距离）
    y_offset = height // 2

    # 判断是否成功检测到线（兼容 None 和空数组的情况）
    has_yellow = yellow_lane is not None and len(yellow_lane) > 0
    has_white = white_lane is not None and len(white_lane) > 0

    # 分情况计算横向偏移量 x_offset
    if has_yellow and has_white:
        # 情况1：双线可见 - 最稳定的状态
        # 取两条线在上半部分端点的 x 坐标（x2是较远的端点）
        _, _, left_x2, _ = yellow_lane
        _, _, right_x2, _ = white_lane
        # 计算两线中点，并与画面中心对比，求出偏移量
        mid_x = (left_x2 + right_x2) / 2
        center_x = width / 2
        x_offset = mid_x - center_x
    elif has_yellow:
        # 情况2：仅见左黄线 - 小车可能偏右，需向左修
        # 优化逻辑：使用线段中心点偏离画面中心的距离，比单纯用 x2-x1 更稳定
        x1, _, x2, _ = yellow_lane
        lane_center_x = (x1 + x2) / 2
        # 假设黄线理想位置在画面（左侧）1/3 处
        target_x = width / 3
        x_offset = lane_center_x - target_x
    elif has_white:
        # 情况3：仅见右白线 - 小车可能偏左，需向右修
        x1, _, x2, _ = white_lane
        lane_center_x = (x1 + x2) / 2
        # 假设白线理想位置在画面（右侧）2/3 处
        target_x = width * 2 / 3
        x_offset = lane_center_x - target_x
    else:
        # 情况4：无线可见 - 丢失目标
        print("检测不到车道线，即将停止或保持上一次状态")
        # 使用 None 表示无效状态，比 -1 更清晰
        return None

    # 计算最终转向角度
    # 通过反正切函数计算偏移角度（弧度），角度变化与偏移距离成非线性关系，轻微偏离时微调，严重偏离时大调
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    # 将弧度转换为角度
    angle_to_mid_deg = angle_to_mid_radian * 180.0 / math.pi
    # 归一化处理，将角度映射到 [-1.0, 1.0] 的区间，使用 np.clip 限制范围，防止异常值溢出
    steering_angle = np.clip(angle_to_mid_deg / 45.0, -1.0, 1.0)

    return steering_angle

