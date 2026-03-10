import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import time
import tracemalloc
import pandas as pd
import math
import os

# ================= 参数配置 =================
VIDEO_PATH = 'bridge.avi'  # 你的视频路径
MAG_FACTOR = 100.0  # 放大倍数
ROI_POINT = (271, 80)  # (x, y)214, 887  400 200   271, 80
SLICE_X = 271  # 切片位置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
startframe=7001,endframe=9000

# ===========================================

def calculate_metrics(img_pred, img_gt):
    img1 = img_pred.astype(np.float64)
    img2 = img_gt.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 0.0, 100.0
    rmse = np.sqrt(mse)
    psnr = 20 * math.log10(255.0 / rmse)
    return rmse, psnr


# 通用的 OpenCV 变形函数 (保证成功率)
def warp_with_opencv(img, flow, factor):
    h, w = img.shape[:2]
    # 生成网格
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # 放大光流并应用反向变形
    # map_x = x - factor * u
    map_x = (grid_x - factor * flow[..., 0]).astype(np.float32)
    map_y = (grid_y - factor * flow[..., 1]).astype(np.float32)

    # 执行重映射
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


class MotionMagnifier:
    def __init__(self, name):
        self.name = name
        self.displacements = []
        self.slices = []
        self.psnr_list = []
        self.rmse_list = []
        self.exec_time = 0
        self.peak_memory = 0

    def process_video(self, video_path, start_frame=0, end_frame=None):
        cap = cv2.VideoCapture(video_path)

        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 处理帧范围（按包含 end_frame）
        if end_frame is None:
            end_frame = total_frames - 1
        end_frame = min(end_frame, total_frames - 1)

        if start_frame < 0 or start_frame >= total_frames:
            print(f"❌ start_frame 越界: {start_frame}, 总帧数={total_frames}")
            cap.release()
            return
        if end_frame < start_frame:
            print(f"❌ end_frame({end_frame}) < start_frame({start_frame})")
            cap.release()
            return

        # 延迟初始化 VideoWriter
        out = None
        vid_name = f"{self.name}_magnified_x{int(MAG_FACTOR)}_{start_frame}-{end_frame}.avi"

        # 跳到起始帧，读取参考帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, ref_frame = cap.read()
        if not ret:
            print("❌ 无法读取参考帧")
            cap.release()
            return

        # 准备模型
        self.prepare(ref_frame)

        # 重置数据（避免重复调用时累积）
        self.displacements = []
        self.slices = []
        self.psnr_list = []
        self.rmse_list = []
        self.exec_time = 0
        self.peak_memory = 0

        # 数据初始化（参考帧）
        self.displacements.append(0)
        self.psnr_list.append(100.0)
        self.rmse_list.append(0.0)
        self.slices.append(ref_frame[:, SLICE_X, :])

        tracemalloc.start()
        start_t = time.time()

        first_frame_ready = ref_frame
        frame_idx = start_frame + 1

        print(f"[{self.name}] 处理帧范围: {start_frame} ~ {end_frame}")

        while True:
            if frame_idx > end_frame:
                break

            ret, curr_frame = cap.read()
            if not ret:
                break

            # 1. 计算光流
            flow = self.compute_flow(ref_frame, curr_frame)

            if frame_idx == start_frame + 1:
                print(f"\n[{self.name}] Debug: Max Flow={np.max(np.abs(flow)):.4f} px")

            # 2. 提取位移
            u, v = flow[ROI_POINT[1], ROI_POINT[0]]
            self.displacements.append(np.sqrt(u ** 2 + v ** 2))

            # 3. 运动放大
            mag_frame = warp_with_opencv(ref_frame, flow, MAG_FACTOR)

            # 初始化写入器
            if out is None:
                h_out, w_out = mag_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(vid_name, fourcc, fps, (w_out, h_out))
                out.write(first_frame_ready)  # 参考帧

            out.write(mag_frame)
            self.slices.append(mag_frame[:, SLICE_X, :])

            rmse, psnr = calculate_metrics(mag_frame, curr_frame)
            self.rmse_list.append(rmse)
            self.psnr_list.append(psnr)

            if (frame_idx - start_frame) % 10 == 0:
                print(f"\r[{self.name}] Frame {frame_idx}/{end_frame}", end="")

            frame_idx += 1

        self.exec_time = time.time() - start_t
        _, self.peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cap.release()
        if out:
            out.release()

        # 保存切片
        slice_img = np.array(self.slices).transpose(1, 0, 2)
        cv2.imwrite(f"{self.name}_slice_{start_frame}-{end_frame}.png", slice_img)

        print(f"\n[{self.name}] 完成! 视频保存在: {os.path.abspath(vid_name)}")

    def prepare(self, ref_frame):
        pass

    def compute_flow(self, ref, curr):
        pass


class DISMagnifier(MotionMagnifier):
    def __init__(self):
        super().__init__("DIS")
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def prepare(self, ref_frame):
        self.prev_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    def compute_flow(self, ref, curr):
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        # 返回 Numpy (H, W, 2)
        return self.dis.calc(self.prev_gray, curr_gray, None)


class RAFTMagnifier(MotionMagnifier):
    def __init__(self):
        super().__init__("RAFT")
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights, progress=False).to(DEVICE).eval()
        self.transforms = weights.transforms()
        self.orig_h = 0
        self.orig_w = 0

    def pad_to_8(self, tensor):
        h, w = tensor.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        return tensor

    def prepare(self, ref_frame):
        self.orig_h, self.orig_w = ref_frame.shape[:2]
        # 这里的输入必须是 uint8 (0-255)，transforms 会自己处理归一化
        self.ref_tensor = torch.from_numpy(ref_frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        self.ref_padded = self.pad_to_8(self.ref_tensor)

    def compute_flow(self, ref, curr):
        curr_tensor = torch.from_numpy(curr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        curr_padded = self.pad_to_8(curr_tensor)

        # 预处理 (自动处理 0-255 到 0-1)
        img1, img2 = self.transforms(self.ref_padded, curr_padded)

        with torch.no_grad():
            flow_padded = self.model(img1, img2)[-1]

        # 裁切回原尺寸
        flow_original = flow_padded[0, :, :self.orig_h, :self.orig_w]

        # 转回 Numpy (H, W, 2)，以此兼容 OpenCV 的 Warping
        return flow_original.permute(1, 2, 0).cpu().numpy()


if __name__ == "__main__":
    # 如果没视频，生成一个测试用的
    if not os.path.exists(VIDEO_PATH):
        print("生成测试视频...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(VIDEO_PATH, fourcc, 30, (261, 221))
        for i in range(60):
            img = np.zeros((221, 261, 3), dtype=np.uint8)
            # 画一个移动的白点
            x = 130 + int(3 * np.sin(i * 0.2))
            cv2.circle(img, (x, 110), 20, (255, 255, 255), -1)
            out.write(img)
        out.release()

    dis = DISMagnifier()
    dis.process_video(VIDEO_PATH,start_frame=startframe,end_frame=endframe)

    raft = RAFTMagnifier()
    raft.process_video(VIDEO_PATH,start_frame=startframe,end_frame=endframe)

    # 保存数据
    min_len = min(len(dis.psnr_list), len(raft.psnr_list))
    df = pd.DataFrame({
        'Frame': range(min_len),
        'DIS_Disp': dis.displacements[:min_len],
        'RAFT_Disp': raft.displacements[:min_len],
        'DIS_RMSE': dis.rmse_list[:min_len],
        'RAFT_RMSE': raft.rmse_list[:min_len]
    })
    df.to_csv("metrics.csv", index=False)
    print("全部完成！")