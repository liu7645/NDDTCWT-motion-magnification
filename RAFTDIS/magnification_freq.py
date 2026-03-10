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
from scipy import signal

# ================= Parameter Configuration =================
VIDEO_PATH = '../data/Bridge.avi'          # Your video path
MAG_FACTOR = 400.0                         # Magnification factor
ROI_POINT = (271, 80)                      # (x, y) coordinates
SLICE_X = 271
SLICE_Y_START = 40
SLICE_Y_END = 140                          # If you want to include =100, 250
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Process frame segment (inclusive of endpoints)
START_FRAME = 1
END_FRAME = 1000

# Frequency selection (Hz)
# For example, bridge frequency band 0.8~1.2 Hz
# If no frequency filtering is desired, set to None
BANDPASS_HZ = (1.6, 1.8)
FILTER_ORDER = 4

# Output encoding (AVI)
FOURCC = 'XVID'

# ===========================================================


def calculate_metrics(img_pred, img_gt):
    """Note: The RMSE/PSNR here only represent the difference from the original frame, not a "magnification accuracy" metric."""
    img1 = img_pred.astype(np.float64)
    img2 = img_gt.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0.0, 100.0
    rmse = np.sqrt(mse)
    psnr = 20 * math.log10(255.0 / rmse)
    return rmse, psnr


def warp_with_opencv(img, flow, factor):
    """Use OpenCV remap for optical flow-based image warping (backward mapping)"""
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # map_x = x - factor * u, map_y = y - factor * v
    map_x = (grid_x - factor * flow[..., 0]).astype(np.float32)
    map_y = (grid_y - factor * flow[..., 1]).astype(np.float32)

    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


class MotionMagnifier:
    def __init__(self, name, bandpass_hz=None, filter_order=4):
        self.name = name
        self.bandpass_hz = bandpass_hz
        self.filter_order = filter_order

        self.displacements = []
        self.slices = []
        self.psnr_list = []
        self.rmse_list = []
        self.exec_time = 0
        self.peak_memory = 0

    def prepare(self, ref_frame):
        """Subclass implementation: prepare model or cache reference frame"""
        pass

    def compute_flow(self, ref, curr):
        """Subclass implementation: return HxWx2 optical flow (np.float32/float64)"""
        pass

    def _validate_points(self, frame):
        h, w = frame.shape[:2]
        x, y = ROI_POINT
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"ROI_POINT={ROI_POINT} out of bounds, video size is (w={w}, h={h})")
        if not (0 <= SLICE_X < w):
            raise ValueError(f"SLICE_X={SLICE_X} out of bounds, video width is {w}")

    def _bandpass_filter_flows(self, flows, fps):
        """
        Apply band-pass filtering to flows (T,H,W,2) along the time axis (axis=0).
        Returns an array of the same shape.
        """
        if self.bandpass_hz is None:
            print(f"[{self.name}] Frequency filtering not enabled (BANDPASS_HZ=None), using original optical flow directly.")
            return flows

        f_low, f_high = self.bandpass_hz
        nyq = 0.5 * fps

        if not (0 < f_low < f_high < nyq):
            raise ValueError(
                f"[{self.name}] Invalid frequency band {self.bandpass_hz} Hz, must satisfy 0 < low < high < fps/2 = {nyq:.3f} Hz"
            )

        T = flows.shape[0]
        print(f"[{self.name}] Applying band-pass filter to optical flow: {f_low:.4f}~{f_high:.4f} Hz, order={self.filter_order}, T={T}, fps={fps:.3f}")

        sos = signal.butter(
            self.filter_order,
            [f_low, f_high],
            btype='bandpass',
            fs=fps,
            output='sos'
        )

        flows_filt = np.empty_like(flows, dtype=np.float32)

        # Filter u/v components separately
        for c in range(2):
            # Prefer zero-phase filtering (better results)
            try:
                tmp = signal.sosfiltfilt(sos, flows[..., c], axis=0)
            except ValueError as e:
                print(f"[{self.name}] ⚠️ sosfiltfilt failed (usually due to too few frames): {e}")
                print(f"[{self.name}] Switching to sosfilt (has phase delay, but works).")
                tmp = signal.sosfilt(sos, flows[..., c], axis=0)

            flows_filt[..., c] = tmp.astype(np.float32)

        # Reference frame corresponds to t=0, force to zero flow to avoid boundary filtering causing the first frame to "warp" too
        flows_filt[0, ...] = 0.0
        return flows_filt

    def process_video(self, video_path, start_frame=0, end_frame=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return

        # Video information
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            print("❌ Invalid video FPS")
            cap.release()
            return

        # Frame range (inclusive of endpoints)
        if end_frame is None:
            end_frame = total_frames - 1
        end_frame = min(end_frame, total_frames - 1)

        if start_frame < 0 or start_frame >= total_frames:
            print(f"❌ start_frame out of bounds: {start_frame}, total frames={total_frames}")
            cap.release()
            return
        if end_frame < start_frame:
            print(f"❌ end_frame({end_frame}) < start_frame({start_frame})")
            cap.release()
            return

        # Output filenames
        if self.bandpass_hz is None:
            suffix = f"fullband_{start_frame}-{end_frame}"
        else:
            suffix = f"bp_{self.bandpass_hz[0]:.3f}-{self.bandpass_hz[1]:.3f}Hz_{start_frame}-{end_frame}"
        vid_name = f"{self.name}_magnified_x{int(MAG_FACTOR)}_{suffix}.avi"
        slice_name = f"{self.name}_slice_{suffix}.png"

        # Jump to start frame and read reference frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, ref_frame = cap.read()
        if not ret:
            print("❌ Cannot read reference frame")
            cap.release()
            return

        try:
            self._validate_points(ref_frame)
        except ValueError as e:
            print(f"❌ {e}")
            cap.release()
            return

        # Model preparation (reference frame)
        self.prepare(ref_frame)

        # Reset records (to avoid accumulation from repeated calls)
        self.displacements = []
        self.slices = []
        self.psnr_list = []
        self.rmse_list = []
        self.exec_time = 0
        self.peak_memory = 0

        print(f"\n[{self.name}] Starting processing")
        print(f"[{self.name}] Video: {video_path}")
        print(f"[{self.name}] Resolution: {width}x{height}, FPS={fps:.3f}, Total frames={total_frames}")
        print(f"[{self.name}] Frame segment: {start_frame} ~ {end_frame} (inclusive)")
        print(f"[{self.name}] Magnification factor: x{MAG_FACTOR}")
        print(f"[{self.name}] Frequency band: {self.bandpass_hz if self.bandpass_hz is not None else 'None (Full band)'}")

        # ---------- Stage 1: Cache frames + Compute optical flow for the whole segment ----------
        # In order to do temporal filtering, the entire flow segment must be cached first
        frames = [ref_frame.copy()]  # Corresponds to t=0
        h, w = ref_frame.shape[:2]
        flows = [np.zeros((h, w, 2), dtype=np.float32)]  # Set t=0 flow to zero (relative to reference frame)

        # Displacement record t=0
        self.displacements.append(0.0)

        tracemalloc.start()
        t_start = time.time()

        frame_idx = start_frame + 1
        while frame_idx <= end_frame:
            ret, curr_frame = cap.read()
            if not ret:
                print(f"\n[{self.name}] ⚠️ Failed to read at frame {frame_idx}, ending early.")
                break

            flow = self.compute_flow(ref_frame, curr_frame).astype(np.float32)  # HxWx2

            if frame_idx == start_frame + 1:
                print(f"[{self.name}] Debug: Max |Flow| = {np.max(np.abs(flow)):.6f} px")

            # Record displacement (original unfiltered displacement, for reference)
            u, v = flow[ROI_POINT[1], ROI_POINT[0]]
            self.displacements.append(float(np.sqrt(u ** 2 + v ** 2)))

            flows.append(flow)
            frames.append(curr_frame.copy())

            if (frame_idx - start_frame) % 10 == 0:
                print(f"\r[{self.name}] Caching optical flow... Frame {frame_idx}/{end_frame}", end="")
            frame_idx += 1

        print()

        cap.release()

        # Stack
        flows = np.stack(flows, axis=0)    # [T, H, W, 2]
        T = flows.shape[0]
        if T < 2:
            print(f"❌ Insufficient valid frames (T={T}), cannot generate magnified video.")
            tracemalloc.stop()
            return

        # ---------- Stage 2: Band-pass filtering along the time axis (optional) ----------
        try:
            flows_used = self._bandpass_filter_flows(flows, fps)
        except Exception as e:
            print(f"❌ Optical flow frequency filtering failed: {e}")
            tracemalloc.stop()
            return

        # Optional: Record "filtered ROI displacement magnitude" for comparison (if you want to see displacement after frequency selection)
        # Overwriting with filtered magnitude here, more in line with "frequency selected magnification" results
        self.displacements = []
        for t in range(T):
            u, v = flows_used[t, ROI_POINT[1], ROI_POINT[0], :]
            self.displacements.append(float(np.sqrt(u ** 2 + v ** 2)))

        # ---------- Stage 3: Generate magnified video using filtered optical flow ----------
        out = None
        self.slices = []
        self.psnr_list = []
        self.rmse_list = []

        # Write reference frame as is
        first_frame = frames[0]
        self.slices.append(first_frame[SLICE_Y_START:SLICE_Y_END, SLICE_X, :])
        self.rmse_list.append(0.0)
        self.psnr_list.append(100.0)

        for t in range(1, T):
            curr_frame = frames[t]
            flow_t = flows_used[t]

            mag_frame = warp_with_opencv(first_frame, flow_t, MAG_FACTOR)

            if out is None:
                h_out, w_out = mag_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*FOURCC)
                out = cv2.VideoWriter(vid_name, fourcc, fps, (w_out, h_out))
                out.write(first_frame)  # Write reference frame

            out.write(mag_frame)
            self.slices.append(mag_frame[SLICE_Y_START:SLICE_Y_END, SLICE_X, :])

            rmse, psnr = calculate_metrics(mag_frame, curr_frame)
            self.rmse_list.append(rmse)
            self.psnr_list.append(psnr)

            if t % 10 == 0:
                abs_frame_idx = start_frame + t
                print(f"\r[{self.name}] Writing video... Frame {abs_frame_idx}/{start_frame + T - 1}", end="")

        print()

        self.exec_time = time.time() - t_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        self.peak_memory = peak_bytes
        tracemalloc.stop()

        if out is not None:
            out.release()

        # Save slice image
        try:
            slice_img = np.array(self.slices).transpose(1, 0, 2)
            cv2.imwrite(slice_name, slice_img)
        except Exception as e:
            print(f"[{self.name}] ⚠️ Failed to save slice image: {e}")

        print(f"[{self.name}] Done!")
        print(f"[{self.name}] Output video: {os.path.abspath(vid_name)}")
        print(f"[{self.name}] Slice image:   {os.path.abspath(slice_name)}")
        print(f"[{self.name}] Total time: {self.exec_time:.2f}s, tracemalloc peak: {self.peak_memory / 1024 / 1024:.2f} MB")


class DISMagnifier(MotionMagnifier):
    def __init__(self, bandpass_hz=None, filter_order=4):
        super().__init__("DIS", bandpass_hz=bandpass_hz, filter_order=filter_order)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.ref_gray = None

    def prepare(self, ref_frame):
        self.ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    def compute_flow(self, ref, curr):
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        # Returns HxWx2
        return self.dis.calc(self.ref_gray, curr_gray, None)


class RAFTMagnifier(MotionMagnifier):
    def __init__(self, bandpass_hz=None, filter_order=4):
        super().__init__("RAFT", bandpass_hz=bandpass_hz, filter_order=filter_order)
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights, progress=False).to(DEVICE).eval()
        self.transforms = weights.transforms()
        self.orig_h = 0
        self.orig_w = 0
        self.ref_padded = None

    def pad_to_8(self, tensor):
        h, w = tensor.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        return tensor

    def prepare(self, ref_frame):
        self.orig_h, self.orig_w = ref_frame.shape[:2]

        # OpenCV uses BGR, convert to RGB before feeding to torchvision RAFT (more standard)
        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        # transforms() expects uint8 [0,255] Tensor, shape [1,C,H,W]
        ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        self.ref_padded = self.pad_to_8(ref_tensor)

    def compute_flow(self, ref, curr):
        curr_rgb = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
        curr_tensor = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        curr_padded = self.pad_to_8(curr_tensor)

        # Preprocessing (weights come with transforms)
        img1, img2 = self.transforms(self.ref_padded, curr_padded)

        with torch.no_grad():
            flow_padded = self.model(img1, img2)[-1]  # [1,2,H,W]

        # Crop back to original size
        flow_original = flow_padded[0, :, :self.orig_h, :self.orig_w]  # [2,H,W]

        # Convert back to HxWx2
        return flow_original.permute(1, 2, 0).cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    # If no video exists, generate a test video (can be deleted)
    if not os.path.exists(VIDEO_PATH):
        print("Generating test video...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(VIDEO_PATH, fourcc, 30, (261, 221))
        for i in range(600):
            img = np.zeros((221, 261, 3), dtype=np.uint8)
            x = 130 + int(3 * np.sin(i * 2 * np.pi * 1.0 / 30.0))  # Approx. 1Hz
            cv2.circle(img, (x, 110), 20, (255, 255, 255), -1)
            out.write(img)
        out.release()

    print("========================================")
    print("Frequency Selected Motion Magnification (RAFT + DIS)")
    print("========================================")
    print(f"VIDEO_PATH   = {VIDEO_PATH}")
    print(f"MAG_FACTOR   = {MAG_FACTOR}")
    print(f"ROI_POINT    = {ROI_POINT}")
    print(f"SLICE_X      = {SLICE_X}")
    print(f"START_FRAME  = {START_FRAME}")
    print(f"END_FRAME    = {END_FRAME}")
    print(f"BANDPASS_HZ  = {BANDPASS_HZ}")
    print(f"FILTER_ORDER = {FILTER_ORDER}")
    print(f"DEVICE       = {DEVICE}")
    print("========================================\n")

    # DIS
    dis = DISMagnifier(bandpass_hz=BANDPASS_HZ, filter_order=FILTER_ORDER)
    dis.process_video(VIDEO_PATH, start_frame=START_FRAME, end_frame=END_FRAME)

    # RAFT
    raft = RAFTMagnifier(bandpass_hz=BANDPASS_HZ, filter_order=FILTER_ORDER)
    raft.process_video(VIDEO_PATH, start_frame=START_FRAME, end_frame=END_FRAME)



    print("\nAll done!")