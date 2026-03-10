import argparse
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import save_image

import cv2
import os
from scipy import signal

from dataset import TrainingFramesDataset, FramesDataset
from test_time_adapt import test_time_adapt
from myutils import get_our_model, write_video, dist_transform


# ============================================================
# 0) Parameter Area (Modify directly here)
# ============================================================

FORCED_ARGS = [
    'inference.py',
    '--config', 'configs/alpha16.color10.yaml',
    '--frames_dir', 'data/Bridge',
    '--resume', 'checkpoints/raft_chkpt_00140.pth',
    '--save_name', 'resultBridge',
    '--alpha', '200',          # Recommend trying 6~12 first, then gradually increase
    '--output_video',
    # '--test_time_adapt',    # Uncomment if TTA is needed
]

# ---- Frequency Selection (Hz) ----
# Set to None if no filtering is desired
BANDPASS_HZ = (1.6, 1.8)
FILTER_ORDER = 4
FPS = 30.0   # Actual frame rate

# ---- Specify Frame Segment (inclusive) ----
START_FRAME = 1
END_FRAME = 1000  # None means up to the last frame

# ---- Bridge mask (PNG grayscale image) ----
MASK_PNG_PATH = 'data/my_mask_for_bridge.png'

# ---- x-t Slice Parameters (fixed x, limited y range) ----
SAVE_XT_SLICE = True
SLICE_X = 887
SLICE_Y_START = 180
SLICE_Y_END = 320

# ---- Slice Image Pixel Scale Bar (MATLAB style, vertical white bar in bottom right) ----
ADD_SCALE_BAR_TO_SLICE = True
SLICE_SCALE_BAR_LEN_PX = 20        # Scale bar length (pixels)
SLICE_SCALE_BAR_LABEL = '20 px'    # Scale bar text
SLICE_SCALE_BAR_MARGIN_PX = 5      # Margin from bottom right corner
SLICE_SCALE_BAR_LINE_THICKNESS = 4
SLICE_SCALE_BAR_TEXT_SCALE = 0.7   # cv2 font scale
SLICE_SCALE_BAR_TEXT_THICKNESS = 2
SLICE_SCALE_BAR_TEXT_GAP_PX = 6    # Horizontal gap between text and scale bar


# ============================================================
# 1) Utility Functions
# ============================================================

def get_model_training_status(model):
    """Compatible with both model / model.module situations"""
    if hasattr(model, 'module') and hasattr(model.module, 'get_training_status'):
        return model.module.get_training_status()
    if hasattr(model, 'get_training_status'):
        return model.get_training_status()
    return False


def format_bp_tag(bandpass_hz):
    if bandpass_hz is None:
        return "fullband"
    f_low, f_high = bandpass_hz
    return f"bp_{f_low:.3f}-{f_high:.3f}Hz"


def tensor_list_to_numpy_tcHW(results):
    """list[tensor(C,H,W) or (1,C,H,W)] -> np[T,C,H,W]"""
    arr = []
    for x in results:
        if x.dim() == 4:
            x = x.squeeze(0)
        arr.append(x.detach().cpu().numpy())
    return np.stack(arr, axis=0).astype(np.float32)


def numpy_tcHW_to_tensor_list(arr):
    return [torch.from_numpy(arr[t]) for t in range(arr.shape[0])]


def temporal_bandpass_on_flowmag_results(results,
                                         fps,
                                         bandpass_hz=(0.8, 1.2),
                                         filter_order=4,
                                         pixels_per_chunk=1024,
                                         prefer_zero_phase=True):
    """
    Apply 'reference frame residual' temporal band-pass filtering to FlowMag output (memory-friendly chunked version)
    results[0] is treated as the reference frame
    """
    if bandpass_hz is None:
        out = []
        for x in results:
            out.append(x.squeeze(0).cpu() if x.dim() == 4 else x.cpu())
        return out

    f_low, f_high = bandpass_hz
    nyq = 0.5 * fps
    if not (0 < f_low < f_high < nyq):
        raise ValueError(f"Invalid frequency band {bandpass_hz}, must satisfy 0 < low < high < fps/2={nyq:.3f}")

    arr = []
    for x in results:
        if x.dim() == 4:
            x = x.squeeze(0)
        arr.append(x.detach().cpu().numpy().astype(np.float32, copy=False))
    arr = np.stack(arr, axis=0).astype(np.float32, copy=False)  # [T,C,H,W]

    T, C, H, W = arr.shape
    if T < 4:
        print("⚠️ Too few frames, skipping frequency filtering")
        return [torch.from_numpy(arr[t]) for t in range(T)]

    ref = arr[0:1]           # [1,C,H,W]
    residual = arr - ref     # [T,C,H,W]

    print(f"Applying temporal band-pass on FlowMag outputs: {bandpass_hz} Hz "
          f"(order={filter_order}, fps={fps}, T={T}, HxW={H}x{W}, chunk={pixels_per_chunk})")

    sos = signal.butter(filter_order, [f_low, f_high], btype='bandpass', fs=fps, output='sos')
    sos = np.asarray(sos, dtype=np.float32)

    residual_bp = np.empty_like(residual, dtype=np.float32)
    n_pix = H * W

    for c in range(C):
        x = residual[:, c, :, :].reshape(T, n_pix)  # [T, H*W]
        y = np.empty_like(x, dtype=np.float32)

        print(f"  Channel {c+1}/{C}: filtering {n_pix} pixels in chunks...")
        n_chunks = (n_pix + pixels_per_chunk - 1) // pixels_per_chunk
        progress_step = max(n_chunks // 10, 1)

        for chunk_idx, s in enumerate(range(0, n_pix, pixels_per_chunk)):
            e = min(s + pixels_per_chunk, n_pix)
            block = np.ascontiguousarray(x[:, s:e], dtype=np.float32)

            if prefer_zero_phase:
                try:
                    y_block = signal.sosfiltfilt(sos, block, axis=0)
                except Exception as err:
                    if chunk_idx == 0:
                        print(f"    ⚠️ sosfiltfilt failed, using sosfilt instead (with phase delay). Reason: {err}")
                    y_block = signal.sosfilt(sos, block, axis=0)
            else:
                y_block = signal.sosfilt(sos, block, axis=0)

            y[:, s:e] = np.asarray(y_block, dtype=np.float32)

            if (chunk_idx % progress_step == 0) or (chunk_idx == n_chunks - 1):
                print(f"    chunk {chunk_idx+1}/{n_chunks}", end='\r')

        print()
        residual_bp[:, c, :, :] = y.reshape(T, H, W)

    residual_bp[0] = 0.0  # Force 0 residual for the reference frame
    out = np.clip(ref + residual_bp, 0.0, 1.0).astype(np.float32, copy=False)
    return [torch.from_numpy(out[t]) for t in range(T)]


def load_mask_tensor(frames_dataset,
                     start_frame=0,
                     png_mask_path=None,
                     npy_mask_path=None,
                     soft_mask=0):
    """
    Load and align mask to frame size, return torch.FloatTensor [H,W], value range [0,1] (CPU)
    Priority: PNG > NPY > None
    """
    if len(frames_dataset) == 0:
        return None

    ref_idx = min(max(start_frame, 0), len(frames_dataset) - 1)
    sample = frames_dataset[ref_idx]  # [C,H,W]
    _, H, W = sample.shape
    mask = None

    if png_mask_path is not None and os.path.exists(png_mask_path):
        print(f"Successfully found mask file (PNG): {png_mask_path}, enabling local magnification mode")
        mask_img = cv2.imread(png_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"⚠️ Cannot read {png_mask_path}, falling back to full image")
            return None
        if (mask_img.shape[0] != H) or (mask_img.shape[1] != W):
            print(f"Mask size {mask_img.shape[::-1]} does not match frame size {(W, H)}, resizing automatically")
            mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = torch.tensor(mask_img / 255.0, dtype=torch.float32)

    elif npy_mask_path is not None and os.path.exists(npy_mask_path):
        print(f"Loading NPY mask: {npy_mask_path}")
        mask_np = np.load(npy_mask_path)
        if mask_np.ndim != 2:
            raise ValueError(f"NPY mask must be 2D [H,W], current shape={mask_np.shape}")
        if (mask_np.shape[0] != H) or (mask_np.shape[1] != W):
            print(f"NPY mask size {mask_np.shape[::-1]} does not match frame size {(W, H)}, resizing automatically")
            mask_np = cv2.resize(mask_np.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        mask = torch.tensor(mask_np, dtype=torch.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
    else:
        print("No valid mask file found, will use default full image magnification")
        return None

    if soft_mask and soft_mask > 0:
        print(f"Softening mask (soft_mask={soft_mask})")
        dist = dist_transform(mask)
        dist[dist < soft_mask] = 1
        dist[dist >= soft_mask] = 0
        mask = dist

    return torch.clamp(mask.float(), 0.0, 1.0)


def add_vertical_scale_bar_bgr(img_bgr,
                               scale_len_px=20,
                               label='20 px',
                               margin=5,
                               line_thickness=4,
                               text_scale=0.7,
                               text_thickness=2,
                               text_gap_px=6):
    """
    Add vertical pixel scale bar (white) to the bottom right of the image, MATLAB style.
    Input and output are both BGR uint8 images.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Scale bar position (bottom right)
    x = max(0, min(w - 1, w - margin - 1))
    y_end = max(0, min(h - 1, h - margin - 1))
    y_start = max(0, y_end - int(scale_len_px))

    # Draw vertical white bar
    cv2.line(out, (x, y_start), (x, y_end), (255, 255, 255), thickness=line_thickness, lineType=cv2.LINE_AA)

    # Place text to the left of the scale bar, vertically centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(label)
    (tw, th), baseline = cv2.getTextSize(text, font, text_scale, text_thickness)
    text_x = max(0, x - text_gap_px - tw)
    text_y = int((y_start + y_end) / 2 + th / 2)
    text_y = max(th + 1, min(h - baseline - 1, text_y))

    cv2.putText(out, text, (text_x, text_y), font, text_scale, (255, 255, 255),
                thickness=text_thickness, lineType=cv2.LINE_AA)

    return out


def save_xt_slice_from_results(results_for_save,
                               save_dir,
                               file_stem,
                               slice_x=271,
                               y_start=40,
                               y_end=141,
                               add_scale_bar=True,
                               scale_bar_len_px=20,
                               scale_bar_label='20 px',
                               scale_bar_margin_px=5,
                               scale_bar_line_thickness=4,
                               scale_bar_text_scale=0.7,
                               scale_bar_text_thickness=2,
                               scale_bar_text_gap_px=6):
    """
    Extract x-t slice with fixed x and limited y range from result sequence (list of [C,H,W], RGB, [0,1]).
    Output image size is roughly [y_range, T, 3]:
      - Vertical axis: y
      - Horizontal axis: time frames
    """
    if results_for_save is None or len(results_for_save) == 0:
        print("⚠️ No results available for slice saving")
        return None

    first = results_for_save[0]
    if first.dim() == 4:
        first = first.squeeze(0)
    C, H, W = first.shape

    if not (0 <= slice_x < W):
        raise ValueError(f"SLICE_X={slice_x} out of bounds, image width W={W}")
    if y_end is None:
        y_end = H
    if not (0 <= y_start < y_end <= H):
        raise ValueError(f"Y range out of bounds: [{y_start}, {y_end}), image height H={H}")

    cols = []
    for img in results_for_save:
        if img.dim() == 4:
            img = img.squeeze(0)
        arr_rgb = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)  # [H,W,3] RGB
        col = arr_rgb[y_start:y_end, slice_x, :]  # [h_slice,3]
        cols.append(col)

    # [T,h_slice,3] -> [h_slice,T,3]
    slice_img_rgb = np.stack(cols, axis=0).transpose(1, 0, 2)

    # cv2 uses BGR for writing images
    slice_img_bgr = slice_img_rgb[..., ::-1].copy()

    if add_scale_bar:
        slice_img_bgr = add_vertical_scale_bar_bgr(
            slice_img_bgr,
            scale_len_px=scale_bar_len_px,
            label=scale_bar_label,
            margin=scale_bar_margin_px,
            line_thickness=scale_bar_line_thickness,
            text_scale=scale_bar_text_scale,
            text_thickness=scale_bar_text_thickness,
            text_gap_px=scale_bar_text_gap_px
        )

    out_path = save_dir / f"{file_stem}_slice_x{slice_x}_y{y_start}-{y_end-1}.png"
    cv2.imwrite(str(out_path), slice_img_bgr)
    print(f"saved xt-slice to {out_path}")
    return out_path


# ============================================================
# 2) Core Inference (Frame Segment + Frequency Filtering + Slice)
# ============================================================

def inference(model,
              frames_dataset,
              save_dir,
              alpha=5.0,
              max_alpha=16.0,
              mask=None,
              num_device=1,
              output_video=False,
              fps=30.0,
              bandpass_hz=None,
              filter_order=4,
              start_frame=0,
              end_frame=None,
              save_xt_slice=False,
              slice_x=271,
              slice_y_start=40,
              slice_y_end=141,
              add_scale_bar_to_slice=True,
              slice_scale_bar_len_px=20,
              slice_scale_bar_label='20 px',
              slice_scale_bar_margin_px=5,
              slice_scale_bar_line_thickness=4,
              slice_scale_bar_text_scale=0.7,
              slice_scale_bar_text_thickness=2,
              slice_scale_bar_text_gap_px=6):
    """
    Within the specified frame segment, perform FlowMag magnification using the first frame as reference;
    Optional: Apply temporal band-pass filtering to the reference frame residual of FlowMag output;
    Optional: Save x-t slice image with fixed x/limited y (and add pixel scale bar).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(model, nn.Module):
        model.eval()

    training_status = get_model_training_status(model)

    # ---- Frame Segment Processing ----
    n_total = len(frames_dataset)
    if n_total == 0:
        raise ValueError("frames_dataset is empty")

    if end_frame is None:
        end_frame = n_total - 1
    end_frame = min(end_frame, n_total - 1)

    if start_frame < 0 or start_frame >= n_total:
        raise ValueError(f"start_frame out of bounds: {start_frame}, total frames={n_total}")
    if end_frame < start_frame:
        raise ValueError(f"end_frame({end_frame}) < start_frame({start_frame})")

    print("=" * 60)
    print("FlowMag Inference (Segment + Optional Temporal Band-pass + XT Slice)")
    print(f"Segment frames      : {start_frame} ~ {end_frame} (inclusive)")
    print(f"Alpha               : {alpha}")
    print(f"Model max alpha     : {max_alpha}")
    print(f"FPS                 : {fps}")
    print(f"Bandpass            : {bandpass_hz}")
    print(f"Filter order        : {filter_order}")
    print(f"Output video        : {output_video}")
    print(f"Mask enabled        : {mask is not None}")
    print(f"Save x-t slice      : {save_xt_slice}")
    if save_xt_slice:
        print(f"Slice X             : {slice_x}")
        print(f"Slice Y range       : {slice_y_start}:{slice_y_end-1}")
        print(f"Slice scale bar     : {add_scale_bar_to_slice} ({slice_scale_bar_label})")
    print("=" * 60)

    # ---- Recursive logic for alpha (keeping original author's strategy) ----
    if alpha > max_alpha and np.sqrt(alpha) < max_alpha:
        our_alpha = np.sqrt(alpha)
        num_recursion = 2
    elif alpha <= max_alpha:
        our_alpha = alpha
        num_recursion = 1
    else:
        raise Exception(f'alpha out of range: alpha={alpha}, max_alpha={max_alpha}')

    print(f"Internal alpha      : {our_alpha} (recursion={num_recursion})")

    if mask is not None:
        mask = mask.to(device)

    results = []  # list of [C,H,W] CPU tensors

    with torch.no_grad():
        # Reference frame = first frame of the specified segment
        im0 = frames_dataset[start_frame][None].to(device)  # [1,C,H,W]
        results.append(im0[0].detach().cpu())               # [C,H,W]

        for i in tqdm(range(start_frame + 1, end_frame + 1), desc="FlowMag segment inference"):
            im1 = frames_dataset[i][None].to(device)
            frames = torch.stack([im0, im1], dim=2).repeat(num_device, 1, 1, 1, 1)

            for _ in range(num_recursion):
                if training_status:
                    pred, _, _ = model(frames, alpha=our_alpha, mask=mask)
                else:
                    pred = model(frames, alpha=our_alpha, mask=mask)

                pred_frame = pred[0, :, 0]  # [C,H,W]

                frames = torch.stack(
                    [im0, pred_frame.unsqueeze(0)], dim=2
                ).repeat(num_device, 1, 1, 1, 1)

            results.append(pred_frame.detach().cpu())

    # ---- Frequency Filtering (FlowMag output residual) ----
    if bandpass_hz is not None:
        try:
            results_for_save = temporal_bandpass_on_flowmag_results(
                results,
                fps=fps,
                bandpass_hz=bandpass_hz,
                filter_order=filter_order,
                pixels_per_chunk=1024,  # Can change to 512 if memory is tight
                prefer_zero_phase=True
            )
        except Exception as e:
            print(f"⚠️ Temporal band-pass failed, falling back to original FlowMag output: {e}")
            results_for_save = [x.cpu() for x in results]
    else:
        results_for_save = [x.cpu() for x in results]

    # ---- File Name Tags ----
    range_tag = f"f{start_frame}-{end_frame}"
    bp_tag = format_bp_tag(bandpass_hz)
    alpha_tag = f"x{alpha:g}"
    prefix = "masked_" if mask is not None else ""
    file_stem = f'{prefix}{alpha_tag}_{range_tag}_{bp_tag}'

    # ---- Save Video/Images ----
    if output_video:
        # In original FlowMag code, flip([-1]) is RGB->BGR, suitable for writing video
        saved_frames = [
            (255 * img.permute(1, 2, 0).flip([-1]).numpy()).astype(np.uint8)
            for img in results_for_save
        ]
        video_path = str(save_dir / f'{file_stem}.mp4')
        write_video(saved_frames, fps, video_path)
        print(f'saved the video to {video_path}')
    else:
        img_dir = save_dir / file_stem
        img_dir.mkdir(exist_ok=True, parents=True)
        for i, img in enumerate(results_for_save):
            save_image(img, img_dir / f'{i + 1:04}.png')
        print(f'saved the images to {str(img_dir)}')

    # ---- Save x-t Slice (fixed x, limited y) ----
    if save_xt_slice:
        try:
            save_xt_slice_from_results(
                results_for_save=results_for_save,
                save_dir=save_dir,
                file_stem=file_stem,
                slice_x=slice_x,
                y_start=slice_y_start,
                y_end=slice_y_end,
                add_scale_bar=add_scale_bar_to_slice,
                scale_bar_len_px=slice_scale_bar_len_px,
                scale_bar_label=slice_scale_bar_label,
                scale_bar_margin_px=slice_scale_bar_margin_px,
                scale_bar_line_thickness=slice_scale_bar_line_thickness,
                scale_bar_text_scale=slice_scale_bar_text_scale,
                scale_bar_text_thickness=slice_scale_bar_text_thickness,
                scale_bar_text_gap_px=slice_scale_bar_text_gap_px
            )
        except Exception as e:
            print(f"⚠️ Failed to save x-t slice: {e}")

    return results_for_save


# ============================================================
# 3) Main Program
# ============================================================

if __name__ == '__main__':
    import sys

    sys.argv = FORCED_ARGS

    parser = argparse.ArgumentParser(description='FlowMag inference with optional temporal band-pass + XT slice')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--frames_dir', type=str, required=True, help='path to directory of frames to magnify')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--save_name', type=str, required=True, help='name to save under')
    parser.add_argument('--alpha', type=float, required=True, help='amount to magnify motion')
    parser.add_argument('--mask_path', type=str, default=None, help='path to numpy mask (.npy), optional')
    parser.add_argument('--soft_mask', type=int, default=0, help='how much to soften mask. 0 is none, higher is more')
    parser.add_argument('--output_video', action='store_true')
    parser.add_argument('--test_time_adapt', action='store_true')
    parser.add_argument('--tta_epoch', type=int, default=3, help='number of epochs for test time adaptation')
    args = parser.parse_args()

    # Create dataset
    frames_dataset = TrainingFramesDataset(args.frames_dir) if args.test_time_adapt else FramesDataset(args.frames_dir)

    # Read max_alpha from config
    config = OmegaConf.load(args.config)
    max_alpha = config.train.alpha_high

    # Output directory (keeping original author's style)
    save_dir = Path(args.resume).parent.parent / 'inference' / args.save_name
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create model
    model, epoch = get_our_model(args, args.test_time_adapt)

    # Load mask (PNG priority, then NPY)
    mask_tensor = load_mask_tensor(
        frames_dataset=frames_dataset,
        start_frame=START_FRAME,
        png_mask_path=MASK_PNG_PATH,
        npy_mask_path=args.mask_path,
        soft_mask=args.soft_mask
    )

    # TTA Mode
    if args.test_time_adapt:
        save_dir = save_dir / f'tta_epoch{epoch:03}'
        save_dir.mkdir(exist_ok=True, parents=True)

        def inference_fn(model_for_tta, epoch_now):
            new_save_dir = save_dir / f'tta_epoch{epoch_now:03}'
            new_save_dir.mkdir(exist_ok=True, parents=True)

            inference(
                model_for_tta,
                frames_dataset,
                new_save_dir,
                alpha=args.alpha,
                max_alpha=max_alpha,
                mask=mask_tensor,
                num_device=1,
                output_video=args.output_video,
                fps=FPS,
                bandpass_hz=BANDPASS_HZ,
                filter_order=FILTER_ORDER,
                start_frame=START_FRAME,
                end_frame=END_FRAME,
                save_xt_slice=SAVE_XT_SLICE,
                slice_x=SLICE_X,
                slice_y_start=SLICE_Y_START,
                slice_y_end=SLICE_Y_END,
                add_scale_bar_to_slice=ADD_SCALE_BAR_TO_SLICE,
                slice_scale_bar_len_px=SLICE_SCALE_BAR_LEN_PX,
                slice_scale_bar_label=SLICE_SCALE_BAR_LABEL,
                slice_scale_bar_margin_px=SLICE_SCALE_BAR_MARGIN_PX,
                slice_scale_bar_line_thickness=SLICE_SCALE_BAR_LINE_THICKNESS,
                slice_scale_bar_text_scale=SLICE_SCALE_BAR_TEXT_SCALE,
                slice_scale_bar_text_thickness=SLICE_SCALE_BAR_TEXT_THICKNESS,
                slice_scale_bar_text_gap_px=SLICE_SCALE_BAR_TEXT_GAP_PX
            )

        model, loss_info = test_time_adapt(
            model,
            args.frames_dir,
            num_epochs=args.tta_epoch,
            inference_fn=inference_fn,
            inference_freq=1,
            alpha=None,
            save_dir=save_dir,
            dataset_length=50000
        )

        for loss_name, losses in loss_info.items():
            plt.plot(losses)
            plt.title(loss_name)
            plt.savefig(save_dir / f'{loss_name}.png')
            plt.clf()

    # Normal inference (or inference after TTA)
    inference(
        model,
        frames_dataset,
        save_dir,
        alpha=args.alpha,
        max_alpha=max_alpha,
        mask=mask_tensor,
        num_device=1,
        output_video=args.output_video,
        fps=FPS,
        bandpass_hz=BANDPASS_HZ,
        filter_order=FILTER_ORDER,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        save_xt_slice=SAVE_XT_SLICE,
        slice_x=SLICE_X,
        slice_y_start=SLICE_Y_START,
        slice_y_end=SLICE_Y_END,
        add_scale_bar_to_slice=ADD_SCALE_BAR_TO_SLICE,
        slice_scale_bar_len_px=SLICE_SCALE_BAR_LEN_PX,
        slice_scale_bar_label=SLICE_SCALE_BAR_LABEL,
        slice_scale_bar_margin_px=SLICE_SCALE_BAR_MARGIN_PX,
        slice_scale_bar_line_thickness=SLICE_SCALE_BAR_LINE_THICKNESS,
        slice_scale_bar_text_scale=SLICE_SCALE_BAR_TEXT_SCALE,
        slice_scale_bar_text_thickness=SLICE_SCALE_BAR_TEXT_THICKNESS,
        slice_scale_bar_text_gap_px=SLICE_SCALE_BAR_TEXT_GAP_PX
    )