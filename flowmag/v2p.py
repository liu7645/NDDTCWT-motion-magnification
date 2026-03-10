import cv2
import os

video_path = '..\data\Bridge.avi'
output_dir = 'data\Bridge'

start_frame = 1
end_frame = 1800

os.makedirs(output_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
count = start_frame

print(f"正在飞速提取第 {start_frame} 帧到第 {end_frame} 帧，请稍等...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or count > end_frame:
        break
    cv2.imwrite(f"{output_dir}/{count:04d}.png", frame)
    count += 1

cap.release()
print(f"提取完成！图片已存入 {output_dir} 文件夹。")