import cv2
import numpy as np
import math

# 读取图片，并转为HSV格式
img = cv2.imread("output/test_drive.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 发现存在孤立的噪声点，会影响后续计算，所以使用高斯模糊进行噪声滤除
img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 1)

# 定义数组，代表 HSV 空间中黄色的下界和上界
# 车道中间的黄实线
lower_yellow = np.array([15, 40, 40])
upper_yellow = np.array([45, 255, 255])
# 逐个像素检查 hsv 图像，并返回黑白图片
yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
# 车道中间的白实线
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])
white_mask = cv2.inRange(img_hsv, lower_white, upper_white)
# 保存中间结果
# cv2.imwrite("yellow_mask_gauss.jpg", yellow_mask)
# cv2.imwrite("white_mask_gauss.jpg", white_mask)

# 基于Canny算法的边缘轮廓提取
# 黄实线边缘提取，200和400分别是阈值，一般不用修改
yellow_edge = cv2.Canny(yellow_mask, 200, 400)
# 白实线边缘提取
white_edge = cv2.Canny(white_mask, 200, 400)
# 保存中间结果
# cv2.imwrite("yellow_edge_canny.jpg", yellow_edge)
# cv2.imwrite("white_edge_canny.jpg", white_edge)


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


# 只保留画面左下方的区域（寻找左侧黄线）
yellow_cropped = region_of_interest(yellow_edge, side="left")
# 只保留画面右下方的区域（寻找右侧白线）
white_cropped = region_of_interest(white_edge, side="right")
# cv2.imwrite("yellow_cropped.jpg", yellow_cropped)
# cv2.imwrite("white_cropped.jpg", white_cropped)

# 基于霍夫变换的直线检测
# 距离分辨率，以像素为单位。设为1表示搜索直线时，每次移动1个像素的距离
rho = 1
# 角度分辨率，以弧度为单位。np.pi / 180 等于 1度，表示每次搜索旋转1度
angle = np.pi / 180
# 一条直线至少需要经过多少个像素点才会被确认为有效直线（投票数阈值）
min_thr = 10

# 对黄色区域的裁剪图进行直线检测
#   yellow_cropped: 输入的边缘图像（单通道黑白图）
#   rho, angle, min_thr: 上面定义的基础参数
#   np.array([]): 占位参数，在旧版OpenCV中用于指定输出向量，现在通常传空数组即可
#   minLineLength=8: 能被接受的最短线段长度。小于8个像素的线段将被丢弃（过滤小噪点）
#   maxLineGap=8: 允许连接的最大线段间隙。如果两条断开的线段间隙小于8像素，会被合并成一条线（修复车道线被遮挡断裂的问题）
yellow_lines = cv2.HoughLinesP(yellow_cropped, rho, angle, min_thr, np.array([]), minLineLength=8, maxLineGap=8)
# 对白色区域的裁剪图进行直线检测
white_lines = cv2.HoughLinesP(white_cropped, rho, angle, min_thr, np.array([]), minLineLength=8, maxLineGap=8)

# 输出格式通常为: [[[x1, y1, x2, y2]], [[x1, y1, x2, y2]], ...]，其中 (x1, y1) 是线段起点，(x2, y2) 是线段终点
# # 打印检测到的黄色线段数组
# print("黄色线段坐标：")
# print(yellow_lines)
# # 打印检测到的白色线段数组
# print("白色线段坐标：")
# print(white_lines)


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


# 在进行几何计算和绘图时，依然使用最初的 img
# 聚合左侧黄线（传入 direction="left"）
yellow_lane = average_lines(img, yellow_lines, direction="left")
# 聚合右侧白线（传入 direction="right"）
white_lane = average_lines(img, white_lines, direction="right")
# print("拟合后的左侧黄线坐标:", yellow_lane)
# print("拟合后的右侧白线坐标:", white_lane)


# 可视化检测到的线段
def draw_lane_line(img, lane_line, line_color=(0, 0, 255), line_width=3):
    # 创建一个与原图大小相同的全黑遮罩图层（用于单独画线）
    line_mask = np.zeros_like(img)

    # 如果没有检测到线，直接返回原图
    if lane_line is None or len(lane_line) == 0:
        return img
    # 之前的数学计算可能产生浮点数，而 cv2.line 绝对不接受浮点数坐标
    x1, y1, x2, y2 = map(int, lane_line)

    # 在黑色的遮罩图层上画出实心的线条
    cv2.line(line_mask, (x1, y1), (x2, y2), line_color, line_width)
    # 将原图与线条遮罩进行融合叠加
    # cv2.addWeighted(图1, 权重1, 图2, 权重2, 亮度调节值)
    # img 权重 0.8：原图稍微变暗一点，形成透视感；line_mask 权重 1.0：线条保持原色亮度
    blended_img = cv2.addWeighted(img, 0.8, line_mask, 1.0, 1.0)

    return blended_img


# 分别保存黄线和白线的图片（验证单条线是否准确）
# 左侧黄线用红色 (0, 0, 255)
img_yellow = draw_lane_line(img, yellow_lane, line_color=(0, 0, 255), line_width=3)
# 右侧白线用蓝色 (255, 0, 0)，以示区分
img_white = draw_lane_line(img, white_lane, line_color=(255, 0, 0), line_width=3)
# cv2.imwrite("img_yellow_lane.jpg", img_yellow)
# cv2.imwrite("img_white_lane.jpg", img_white)

# # 将两条线同时画在一张图上（这是自动驾驶最终的常见做法）
# # 先画黄线
# img_combined = draw_lane_line(img, yellow_lane, line_color=(0, 0, 255), line_width=3)
# # 在已经画了黄线的图上，继续画白线
# img_combined = draw_lane_line(img_combined, white_lane, line_color=(255, 0, 0), line_width=3)
# cv2.imwrite("img_lane_combined.jpg", img_combined)


# 动作控制
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


img_height, img_width = img.shape[:2]
# 计算转向角
steer_angle = compute_steer_angle(yellow_lane, white_lane, img_height, img_width)
#
if steer_angle is not None:
    print(f"当前转向指令: {steer_angle:.2f} (负数左转, 正数右转, 0直行)")
else:
    print("丢失车道线，执行安全策略...")
