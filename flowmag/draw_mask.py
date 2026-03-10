import cv2
import numpy as np

# 1. 填入你视频第一帧的准确路径
image_path = 'Bridge/0001.png'
img = cv2.imread(image_path)

if img is None:
    print(f"哎呀，找不到图片 {image_path}，请检查路径对不对！")
    exit()

clone = img.copy()
points = [] # 用来记下你鼠标点过的坐标

# 鼠标点击事件的魔法
def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN: # 鼠标左键点击
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1) # 画个红点
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 255, 0), 2) # 连成绿线
        cv2.imshow("Draw Mask", img)

cv2.namedWindow("Draw Mask")
cv2.setMouseCallback("Draw Mask", draw_polygon)

print("============ 操作说明 ============")
print("1. 鼠标左键沿着【桥的轮廓】点一圈（不需要太精细，大致包住就行）。")
print("2. 画完一圈后，按下键盘上的 'Enter' (回车键) 结束。")
print("==================================")

# 等待你画完
while True:
    cv2.imshow("Draw Mask", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 13 or key == ord("q"): # 13 代表回车键
        break
cv2.destroyAllWindows()

# 开始生成你需要的神级遮罩
if len(points) > 2:
    # 先建一个全黑的底图
    mask = np.zeros(clone.shape[:2], dtype=np.uint8)
    # 把你画的区域填成纯白
    pts = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)

    # 【核心魔法 1：向外扩张 (Dilate)】
    # 让遮罩比你画的桥大一圈。这里的 (30, 30) 是扩大的像素值，可以改。
    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 【核心魔法 2：极其平滑的高斯模糊 (Gaussian Blur)】
    # 让边缘从纯白慢慢过渡到黑。(101, 101) 是模糊强度，必须是奇数，越大越平滑。
    mask = cv2.GaussianBlur(mask, (101, 101), 0)

    # 保存结果
    cv2.imwrite('my_mask.png', mask)
    print("\n太棒了！完美的渐变遮罩 my_mask.png 已经生成在当前目录！")
else:
    print("\n点太少了，连个多边形都没凑齐，没法生成哦。")