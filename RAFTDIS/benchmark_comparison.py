import sys
import os
import cv2
import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import argparse
from collections import OrderedDict

# ================= 配置区域 =================
VIDEO_PATH = '..\\synthetic_vibration_2Hz.avi'   # 刚才生成的视频路径
GT_PATH = '..\\ground_truth_2Hz.txt'             # 刚才生成的真值文件
RAFT_MODEL_PATH = 'raft-small.pth'           # RAFT模型路径
SAVE_CSV_PATH = 'displacement_results.csv'   # 结果保存文件名
PLOT_PATH = 'comparison_plot.png'            # 结果图保存文件名

# 监测点的坐标 (取图像中心，避开边缘)
# 如果之前的视频是 800x600，中心就是 (400, 300)
PROBE_X = 400
PROBE_Y = 300
# ===========================================

# 尝试导入 RAFT 核心模块
# 必须确保脚本在 RAFT-master 根目录下
sys.path.append('core')
try:
    from raft import RAFT
    from utils.utils import InputPadder
except ImportError:
    print("❌ 错误: 找不到 RAFT 模块。请确保本脚本在 'RAFT-master' 文件夹内，且 'core' 文件夹存在。")
    sys.exit(1)


# 定义一个简单的参数类，用于初始化 RAFT
class Args:
    def __init__(self, model_path):
        self.model = model_path
        self.small = True  # 使用 raft-small
        self.mixed_precision = False
        self.alternate_corr = False

        # 补上 RAFT 可能检查的缺失参数，防止报错
        self.dropout = 0
        self.corr_levels = 4  # raft-small 需要这些默认值
        self.corr_radius = 3
        self.epsilon = 1e-8  # 防止除零错误
        self.clip = 1.0  # 梯度裁剪

    # 关键修复：让这个类支持 'if "xxx" in args' 这种写法
    def __contains__(self, key):
        return hasattr(self, key)

def get_memory_usage():
    """获取当前进程的内存占用 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_opencv_dis(video_path):
    print(f"🚀 [OpenCV DIS] 开始运行...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return [], 0, 0

    # 预热一下
    ret, prev_frame = cap.read()
    if not ret: return [], 0, 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 初始化 DIS
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    
    y_displacements = [0.0] # 第一帧位移为0
    
    start_time = time.time()
    start_mem = get_memory_usage()
    peak_mem = start_mem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算光流
        flow = dis.calc(prev_gray, curr_gray, None)
        
        # 提取中心点的 Y 轴位移 (flow[y, x, 1])
        dy = flow[PROBE_Y, PROBE_X, 1]
        
        # 注意：光流是两帧之间的相对位移。
        # 如果要和真值(绝对位置)对比，需要累加？
        # 不，合成视频是相对于第一帧在振动，但光流算法计算的是“当前帧相对于上一帧”的移动(瞬时速度)，
        # 还是“当前帧相对于基准帧”？
        # 通常 DIS/RAFT 计算的是 pair-wise (两帧之间)。
        # 但我们的合成视频是每个时刻相对于 t=0 都有位移。
        # ❌ 不对！光流计算的是 frame_t 和 frame_t+1 之间的像素运动。
        # 我们的真值是位置 y = A*sin(t)。光流算出来的是速度 v ≈ y'。
        # 💡 修正策略：为了简单对比，我们将 RAFT/DIS 的输入改为 (第0帧, 第t帧)。
        # 这样光流输出的直接就是绝对位移，能直接跟 Ground Truth 对比。
        
        # ---- 策略调整：始终计算 当前帧 vs 第一帧 (Frame 0) ----
        # 这样得到的 flow 就是绝对位移
        flow_abs = dis.calc(prev_gray, curr_gray, None) # 这里其实应该用 frame0，但DIS通常用于连续帧
        # 为了严谨，我们这里采用连续帧积分？或者直接对比 frame 0?
        # 考虑到 DIS 的特性，它适合连续小运动。
        # 我们这里改用：每次都计算 Frame 0 和 Frame t 的光流。
        # 为了不破坏上面的循环结构，我们重新读一下 Frame 0
        
        prev_gray = curr_gray # 正常光流是递推的，这里我们先按累加处理，或者看下面 RAFT 的逻辑
    
    # --- 重新实现：为了准确对比绝对位移，我们用 Frame 0 作为 Reference ---
    # DIS 对大位移可能不准，但这里只有0.5像素，应该没问题。
    cap.release()
    
    # === 正式执行逻辑 (Frame 0 vs Frame t) ===
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    
    results = []
    
    # 记录时间从这里重新开始
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算 Frame 0 -> Frame t 的光流
        flow = dis.calc(gray0, gray_curr, None)
        
        dy = flow[PROBE_Y, PROBE_X, 1]
        results.append(dy)
        
        # 监控内存
        current_mem = get_memory_usage()
        if current_mem > peak_mem:
            peak_mem = current_mem
            
    end_time = time.time()
    cap.release()
    
    total_time = end_time - start_time
    mem_used = peak_mem - start_mem # 增量内存
    
    print(f"✅ [OpenCV DIS] 完成。耗时: {total_time:.4f}s, 内存峰值: {peak_mem:.2f}MB")
    return results, total_time, peak_mem


def run_raft(video_path, model_path):
    print(f"🚀 [RAFT] 开始运行 (模型: {model_path})...")

    # 1. 加载模型
    args = Args(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   运行设备: {device}")

    # 如果设备是 cuda，打印显卡名字让你放心
    if device.type == 'cuda':
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 警告: 依然检测不到 GPU，请检查 PyTorch 版本！")

    model = RAFT(args)

    # 加载权重
    state_dict = torch.load(args.model, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()

    # 2. 读取视频
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret: return [], 0, 0, 0

    # 准备第一帧 (Frame 0)
    frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    image0 = torch.from_numpy(frame0_rgb).permute(2, 0, 1).float()[None].to(device)

    padder = InputPadder(image0.shape)

    # === 🛠️ 修复点 1: 强行取列表第0个元素 ===
    # padder.pad 返回的是 list，必须加 [0] 变成 Tensor
    image0_padded = padder.pad(image0)[0]
    # ======================================

    results = []

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_ram = get_memory_usage()
    peak_ram = start_ram

    frame_count = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"   正在处理第 {frame_count} 帧...", end='\r')

            # 准备当前帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_curr = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()[None].to(device)

            # === 🛠️ 修复点 2: 这里也要加 [0] ===
            image_curr_padded = padder.pad(image_curr)[0]
            # =================================

            # 运行 RAFT
            _, flow_up = model(image0_padded, image_curr_padded, iters=12, test_mode=True)

            # 取回结果
            flow_numpy = flow_up[0].permute(1, 2, 0).cpu().numpy()

            dy = flow_numpy[PROBE_Y, PROBE_X, 1]
            results.append(dy)

            curr_ram = get_memory_usage()
            if curr_ram > peak_ram: peak_ram = curr_ram

    end_time = time.time()
    cap.release()
    print("")

    total_time = end_time - start_time

    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"✅ [RAFT] 完成。耗时: {total_time:.4f}s, RAM峰值: {peak_ram:.2f}MB, 显存占用: {gpu_mem:.2f}MB")
    return results, total_time, peak_ram, gpu_mem

def main():
    # 1. 读取真值
    if not os.path.exists(GT_PATH):
        print("错误：找不到真值文件，请先运行生成视频的脚本。")
        return
        
    print("正在读取真值数据...")
    df_gt = pd.read_csv(GT_PATH)
    gt_time = df_gt.iloc[:, 0].values
    gt_disp = df_gt.iloc[:, 1].values
    
    # 2. 运行算法
    disp_opencv, time_opencv, mem_opencv = run_opencv_dis(VIDEO_PATH)
    disp_raft, time_raft, mem_raft, gpu_raft = run_raft(VIDEO_PATH, RAFT_MODEL_PATH)
    
    # 对齐数据长度 (防止差一两帧)
    min_len = min(len(gt_disp), len(disp_opencv), len(disp_raft))
    gt_disp = gt_disp[:min_len]
    gt_time = gt_time[:min_len]
    disp_opencv = disp_opencv[:min_len]
    disp_raft = disp_raft[:min_len]
    
    # 3. 计算误差 (RMSE)
    rmse_opencv = np.sqrt(np.mean((gt_disp - disp_opencv)**2))
    rmse_raft = np.sqrt(np.mean((gt_disp - disp_raft)**2))
    
    print("\n" + "="*40)
    print(f"📊 最终对比结果 (RMSE 越小越好):")
    print(f"OpenCV DIS RMSE: {rmse_opencv:.6f} pixels")
    print(f"RAFT RMSE      : {rmse_raft:.6f} pixels")
    print(f"OpenCV Time    : {time_opencv:.4f} s")
    print(f"RAFT Time      : {time_raft:.4f} s")
    print("="*40)
    
    # 4. 保存数据到 CSV (方便以后画图)
    df_save = pd.DataFrame({
        'Time_s': gt_time,
        'Ground_Truth': gt_disp,
        'OpenCV_DIS': disp_opencv,
        'RAFT': disp_raft
    })
    df_save.to_csv(SAVE_CSV_PATH, index=False)
    print(f"💾 所有数据已保存至: {SAVE_CSV_PATH}")
    
    # 5. 画图展示
    plt.figure(figsize=(12, 6))
    
    # 画真值 (黑线，粗一点)
    plt.plot(gt_time, gt_disp, color='black', linewidth=2, label='Ground Truth', linestyle='--')
    
    # 画 OpenCV (蓝线)
    plt.plot(gt_time, disp_opencv, color='blue', linewidth=1.5, alpha=0.8, label=f'OpenCV DIS (RMSE={rmse_opencv:.4f})')
    
    # 画 RAFT (红线)
    plt.plot(gt_time, disp_raft, color='red', linewidth=1.5, alpha=0.8, label=f'RAFT (RMSE={rmse_raft:.4f})')
    
    plt.title(f'Sub-pixel Displacement Measurement: GT vs RAFT vs OpenCV\n(Time Cost: RAFT {time_raft:.1f}s, OpenCV {time_opencv:.1f}s)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Displacement (pixels)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.show()
    print(f"📈 对比图已保存至: {PLOT_PATH}")

if __name__ == '__main__':
    main()