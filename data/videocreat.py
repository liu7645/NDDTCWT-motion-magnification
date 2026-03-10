import cv2
import numpy as np
import math
import os

# ================= 配置区域 =================
VIDEO_NAME = 'single_circle_blur_0.1.avi'
DATA_NAME = 'ground_truth_0.1.txt'

WIDTH = 800  # 宽
HEIGHT = 600  # 高
FPS = 30  # 帧率
DURATION_SEC = 60  # 时长
AMP = 0.1  # 振幅 
FREQ = 2.0  # 频率 (2 Hz)

CIRCLE_RADIUS = 100  # 圆的半径
BLUR_KSIZE = (0, 0)  # 高斯模糊核大小 (越大越模糊)
BLUR_SIGMA = 0  # 高斯模糊标准差
# ===========================================

print(f"正在生成: {WIDTH}x{HEIGHT}, 振幅: {AMP}px")

# 1. 创建底图 (绘制单个实心圆)
# 为了防止边缘在模糊时受边界影响，我们把圆画在正中间
base_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8) + 50  # 给个灰色背景(50)，不是纯黑，模拟低对比度
center_x, center_y = WIDTH // 2, HEIGHT // 2

# 绘制实心白圆 (颜色 200，留点余地给噪点)
cv2.circle(base_image, (center_x, center_y), CIRCLE_RADIUS, (200), -1)

# 2. [关键] 添加高斯模糊 (Gaussian Blur)
# 这会让 200->50 的跳变变成平滑的梯度，利于亚像素计算
# base_image = cv2.GaussianBlur(base_image, BLUR_KSIZE, BLUR_SIGMA)

# 3. 添加随机噪声 (模拟真实传感器底噪)
noise = np.random.randint(0, 20, (HEIGHT, WIDTH), dtype=np.uint8)
base_image = cv2.add(base_image, noise)

# 打印关键坐标信息
print("=" * 40)
print(f"圆心坐标: ({center_x}, {center_y})")
print(f"半径: {CIRCLE_RADIUS}")
print(f"!!! 推荐测量点 (ROI_POINT) !!!")
print(f"   上边缘: ({center_x}, {center_y - CIRCLE_RADIUS}) -> 即 (400, 200)")
print(f"   下边缘: ({center_x}, {center_y + CIRCLE_RADIUS}) -> 即 (400, 400)")
print("=" * 40)

# 4. 准备视频写入
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用 MJPG 编码，兼容性好
out = cv2.VideoWriter(VIDEO_NAME, fourcc, FPS, (WIDTH, HEIGHT), isColor=False)

# 5. 生成动画
ground_truth_list = []
total_frames = int(DURATION_SEC * FPS)

for i in range(total_frames):
    t = i / FPS
    # 垂直位移公式
    shift_y = AMP * math.sin(2 * math.pi * FREQ * t)

    ground_truth_list.append(f"{t:.4f}, {shift_y:.6f}")

    # 亚像素平移 (只在 Y 轴)
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])

    # 使用双线性插值 (关键：配合高斯模糊实现亚像素效果)
    shifted_frame = cv2.warpAffine(base_image, M, (WIDTH, HEIGHT),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

    out.write(shifted_frame)

    if i % 50 == 0:
        print(f"\r进度: {i}/{total_frames} | 位移: {shift_y:.4f}", end="")

out.release()

# 保存真值
with open(DATA_NAME, 'w') as f:
    f.write("Time(s), Shift_Y(pixels)\n")
    for line in ground_truth_list:
        f.write(line + '\n')

print(f"\n完成！文件已保存至: {os.getcwd()}")