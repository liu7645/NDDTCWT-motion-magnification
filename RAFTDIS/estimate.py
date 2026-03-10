import sys
import os
import cv2
import numpy as np
import torch
import pandas as pd
from scipy import signal
import time
import psutil
from collections import OrderedDict

# ================= 1. 环境设置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'core'))

try:
    from raft import RAFT
    from utils.utils import InputPadder
except ImportError:
    print("❌ 找不到 RAFT 核心文件，请确保脚本在 RAFT-master 文件夹下。")
    sys.exit(1)


class Args:
    def __init__(self, model_path):
        self.model = model_path
        self.small = True
        self.mixed_precision = False
        self.alternate_corr = False
        self.dropout = 0

    def __contains__(self, key): return hasattr(self, key)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# ================= 2. 核心提取类 =================
class MotionExtractor:
    def __init__(self, raft_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 初始化设备: {self.device}")

        # 加载 RAFT
        self.raft = RAFT(Args(raft_model_path))
        state_dict = torch.load(raft_model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.raft.load_state_dict(new_state_dict)
        self.raft.to(self.device)
        self.raft.eval()

        # 初始化 DIS
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def run_raft_only(self, video_path, points, start, end, scale):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end is None: end = total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame0 = cap.read()
        if not ret: return pd.DataFrame()

        frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        img0_torch = torch.from_numpy(frame0_rgb).permute(2, 0, 1).float()[None].to(self.device)
        padder = InputPadder(img0_torch.shape)
        img0_padded = padder.pad(img0_torch)[0]

        results = []
        frame_idx = start

        print(f"   [RAFT] Running frames {start}-{end}...")
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= end: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_curr = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()[None].to(self.device)
            img_curr_padded = padder.pad(img_curr)[0]

            with torch.no_grad():
                _, flow_up = self.raft(img0_padded, img_curr_padded, iters=10, test_mode=True)
            flow_raft = flow_up[0].permute(1, 2, 0).cpu().numpy()

            row = {'Time_Video': (frame_idx - start) / fps}  # 相对时间
            for i, (px, py) in enumerate(points):
                row[f'RAFT_P{i + 1}_Y'] = flow_raft[py, px, 1] * scale
            results.append(row)
            frame_idx += 1
            if frame_idx % 100 == 0: print(f"    Processing {frame_idx}...", end='\r')

        cap.release()
        return pd.DataFrame(results)

    def run_dis_only(self, video_path, points, start, end, scale):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end is None: end = total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame0 = cap.read()
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

        results = []
        frame_idx = start

        print(f"\n   [DIS] Running frames {start}-{end}...")
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= end: break

            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = self.dis.calc(gray0, gray_curr, None)

            row = {'Time_Video': (frame_idx - start) / fps}  # 相对时间
            for i, (px, py) in enumerate(points):
                row[f'DIS_P{i + 1}_Y'] = flow[py, px, 1] * scale
            results.append(row)

            frame_idx += 1
            if frame_idx % 100 == 0: print(f"    Processing {frame_idx}...", end='\r')

        cap.release()
        return pd.DataFrame(results)


# ================= 3. 实验配置 =================
experiments = [
    {
        'name': 'Bridge',
        'video_file': '../data/Bridge.avi',
        'points': [[271, 80]],  # [x, y]
        'frame_range': [0, 999],  # 提取视频的帧范围
        'mm_per_pixel': -24 / 42 * 25.4,  # 物理比例尺

        # 🌟 调整时间轴的核心配置 🌟
        'time_offset': 545.30,  # 视频第一帧对应的绝对起始时间 (秒)
        'crop_time_window': [545.30, 550.685],  # 最终你要导出的时间段

        # 滤波器配置
        'bandpass_cutoffs': [2.4, 2.7],
    },
]


# ================= 4. 主程序 =================
def main():
    if not os.path.exists('raft-small.pth'):
        print("❌ 错误: 找不到 raft-small.pth")
        return

    engine = MotionExtractor('raft-small.pth')

    for exp in experiments:
        print(f"\n{'=' * 20}\n正在处理: {exp['name']}\n{'=' * 20}")

        if not os.path.exists(exp['video_file']):
            print(f"❌ 找不到视频: {exp['video_file']}")
            continue

        # --- 步骤 1: 提取数据 ---
        df_raft = engine.run_raft_only(exp['video_file'], exp['points'], exp['frame_range'][0], exp['frame_range'][1],
                                       exp['mm_per_pixel'])
        df_dis = engine.run_dis_only(exp['video_file'], exp['points'], exp['frame_range'][0], exp['frame_range'][1],
                                     exp['mm_per_pixel'])

        if df_raft.empty or df_dis.empty:
            continue

        # --- 步骤 2: 修正真实时间轴 ---
        print("\n⚙️ 正在生成真实时间轴并对视频数据进行滤波...")
        # 把相对时间 (0, 0.033, 0.066...) 加上 offset 变成真实的绝对时间
        df_raft['Time_Video'] = df_raft['Time_Video'] + exp['time_offset']
        df_dis['Time_Video'] = df_dis['Time_Video'] + exp['time_offset']

        # --- 步骤 3: 滤波处理 ---
        video_fps = 1.0 / (df_raft['Time_Video'].iloc[1] - df_raft['Time_Video'].iloc[0])
        nyq_video = video_fps / 2.0
        b_band_vid, a_band_vid = signal.butter(1, [f / nyq_video for f in exp['bandpass_cutoffs']], btype='bandpass')

        for i in range(len(exp['points'])):
            p_idx = i + 1

            # RAFT 数据滤波 (直接覆盖，不保留 Raw)
            if f'RAFT_P{p_idx}_Y' in df_raft.columns:
                raw_raft = df_raft[f'RAFT_P{p_idx}_Y'].values
                df_raft[f'RAFT_P{p_idx}_Y_Filtered'] = signal.filtfilt(b_band_vid, a_band_vid, signal.detrend(raw_raft))
                df_raft = df_raft.drop(columns=[f'RAFT_P{p_idx}_Y'])

            # DIS 数据滤波 (直接覆盖，不保留 Raw)
            if f'DIS_P{p_idx}_Y' in df_dis.columns:
                raw_dis = df_dis[f'DIS_P{p_idx}_Y'].values
                df_dis[f'DIS_P{p_idx}_Y_Filtered'] = signal.filtfilt(b_band_vid, a_band_vid, signal.detrend(raw_dis))
                df_dis = df_dis.drop(columns=[f'DIS_P{p_idx}_Y'])

        # --- 步骤 4: 合并数据 ---
        df_raft.reset_index(drop=True, inplace=True)
        df_dis.reset_index(drop=True, inplace=True)
        final_df = pd.concat([df_raft, df_dis.drop(columns=['Time_Video'])], axis=1)

        # --- 步骤 5: 截断到指定区间 ---
        t_start, t_end = exp['crop_time_window']
        final_df = final_df[(final_df['Time_Video'] >= t_start) & (final_df['Time_Video'] <= t_end)]

        # --- 步骤 6: 整理列顺序并保存 ---
        cols = final_df.columns.tolist()
        ordered_cols = ['Time_Video']
        for p_idx in range(1, len(exp['points']) + 1):
            if f'RAFT_P{p_idx}_Y_Filtered' in cols: ordered_cols.append(f'RAFT_P{p_idx}_Y_Filtered')
            if f'DIS_P{p_idx}_Y_Filtered' in cols: ordered_cols.append(f'DIS_P{p_idx}_Y_Filtered')

        final_df = final_df[ordered_cols]
        save_csv = f"Result_{exp['name']}_Filtered_Data.csv"
        final_df.to_csv(save_csv, index=False)

        print(f"💾 截取完成！纯滤波数据已保存至: {save_csv}")


if __name__ == '__main__':
    main()