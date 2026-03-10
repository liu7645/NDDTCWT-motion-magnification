# Supplementary Video Materials
Due to the file size limitations of the submission system, the supplementary video materials demonstrating Bridge.avi have been securely hosted on an external platform. Please view or download the video materials from the following link and place them in the data folder:
https://drive.google.com/drive/folders/17-1nDzezPkzW7BNPC1gYzy2yXxk81fhN?usp=drive_link 

---

# Proposed Method: NDDTCWT Implementation Details
To fully address reproducibility and provide transparency into our N-Dimensional Dual-Tree Complex Wavelet Transform (NDDTCWT) pipeline, we detail the core configurations below.

1. Complex Wavelet Transform Settings
Wavelet Filters: The spatial decomposition is performed using standard Dual-Tree Complex Wavelet filters. Based on our implementation, we utilize the Antonini filters (NDAntonB2) for the first stage decomposition and Kingsbury's Q-shift dual-filter trees (NDdualfilt1) for the subsequent stages. This combination ensures near-perfect shift invariance and good directional selectivity across 6 orientations.
Decomposition Levels: For Motion Magnification, the video is decomposed into 4 levels (ht=3 + 1), and phases across all levels are manipulated. For Displacement Estimation, while 4 levels are computed, we specifically utilize the first 2 levels (uselevel=2) across all 6 orientations to solve the overdetermined system for sub-pixel motion constraints, as high-frequency structural vibrations are most prominent in these finer scales.

2. Phase Processing & Reference-Frame Policy
The complex wavelet coefficients are formed as C = Re + i * Im, where the real and imaginary parts correspond to the outputs of the two parallel filter trees. The phase phi is extracted via angle(C). We employ distinct reference-frame policies tailored to the task:
For Displacement Estimation (Frame-to-Frame Integration): To avoid the severe 2*pi spatial phase wrapping issues that occur over long videos, we do not compare the current frame directly to the first frame. Instead, we compute the phase difference between successive frames (i.e., phi_t - phi_{t-1}). The difference is tightly wrapped to the principal range [-pi, pi] using the mathematical operation: delta_phi = mod(pi + phi_t - phi_{t-1}, 2*pi) - pi. These robust inter-frame differences are then temporally accumulated (integrated) to yield the absolute phase shift relative to the start of the video.
For Motion Magnification (Fixed Reference): The magnification pipeline directly computes the phase difference between the current frame and the fixed first frame (phi_t - phi_1). Since the magnified motion is synthesized directly from these phases, using a fixed reference prevents the accumulation of temporal drift (random walk) in the amplified video.

3. Spatial & Temporal Filtering
Displacement Estimation: The extracted displacement signals pass through a 1st-order Butterworth high-pass filter (cutoff at 0.5 Hz) to remove low-frequency thermal drift and integration artifacts. Then, a 1st-order Butterworth bandpass filter (e.g., 2.4 - 2.7 Hz) isolates the specific modal frequency. Spatial smoothing is applied during the motion constraint solving with sigma = 3.
Motion Magnification: A temporal bandpass filter (e.g., 1.6 - 1.8 Hz) is applied directly to the phase differences along the time axis before multiplying by the magnification factor (e.g., alpha = 400). The spatial smoothing parameter is set to sigma = 0 to preserve maximum image sharpness.

4. Reproducing NDDTCWT Results
Point-wise Displacement Estimation: Run the motionestimatetebridge.m script to process the video alongside the provided accelerometer data.
Run the NDDTCWTmag.m script to generate the amplified video using our phase-based NDDTCWT approach.

Overall pipeline:
1. Compute NDDTCWT decomposition.
2. Extract complex coefficients and phase.
3. Compute phase differences (frame-to-frame or fixed reference).
4. Apply temporal bandpass filtering.
5. Convert phase to displacement or amplify phas.
6. Reconstruct motion or estimate displacement signals.

---

# Baseline Implementations: RAFT & DIS Optical Flow
To ensure strict reproducibility and a deterministic evaluation pipeline, we have standardized our optical flow baseline implementations for both displacement estimation and motion magnification.

1. Model Configurations & Checkpoints
RAFT: For both estimation and magnification tasks, we uniformly utilize the official raft_large model provided by torchvision.models.optical_flow. The model is loaded with the default pre-trained weights (Raft_Large_Weights.DEFAULT).
DIS: We use the Dense Inverse Search (DIS) optical flow implementation from OpenCV, configured with the medium preset (cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).

2. Preprocessing & Inference Settings
Color Space: Input video frames are read using OpenCV (BGR). They are converted to RGB before being fed into the RAFT model, and to Grayscale (cv2.COLOR_BGR2GRAY) for the DIS model.
Padding (RAFT): The RAFT architecture requires spatial dimensions to be multiples of 8. We pad the right and bottom edges of the input tensors prior to inference and crop the output flow field back to the original video resolution.
Inference Iterations: During displacement estimation, the RAFT model is set to run for 10 flow updates (num_flow_updates=10) in evaluation mode.

3. Temporal Filtering
Before evaluating displacements or generating magnified videos, the extracted optical flow signals undergo temporal bandpass filtering to isolate the structural frequencies of interest (e.g., the bridge's natural frequency).
For Displacement Estimation: A 1st-order Butterworth bandpass filter is applied using scipy.signal.filtfilt.
For Motion Magnification: A 4th-order Butterworth bandpass filter is applied along the temporal axis using scipy.signal.sosfiltfilt to ensure zero phase distortion.

4. Motion Magnification & Warping Procedure
As requested, here is the exact procedure used to generate magnified videos from the temporally filtered optical flow baselines:
Warping Strategy: We employ explicit backward warping. Let I(x,y) be the reference frame and (u,v) be the extracted and filtered optical flow at a given pixel. The target mapped coordinates are calculated as: map_x = x - factor * u, map_y = y - factor * v.
Interpolation: The backward mapping is executed using OpenCV's cv2.remap function with bilinear interpolation (cv2.INTER_LINEAR).
Hole Handling & Boundary Conditions: To handle out-of-bounds pixels and "holes" generated by large magnification factors, we utilize OpenCV's reflect boundary condition (borderMode=cv2.BORDER_REFLECT).
Note: This standard explicit warping pipeline fundamentally struggles with occlusions and boundaries, which explains the visual tearing artifacts observed in the RAFT/DIS magnified outputs.

Reproducing Baseline Results on the Bridge Video:
Run the estimatelarge script to extract the displacement signals for specific points using RAFT and DIS.
Run the magnification_freq script to generate the magnified video and spatial-temporal slice images based on the specified frequency band.

Overall baseline pipeline:
1. Read video frames.
2. Compute optical flow using RAFT or DIS.
3. Extract displacement signals.
4. Apply temporal bandpass filtering.
5. Use filtered flow for displacement estimation or motion magnification via backward warping.

---

# Flowmag Reproduction Guide
This directory contains the implementation and necessary scripts to reproduce the Flowmag (Self-Supervised Motion Magnification by Backpropagating Through Optical Flow) baseline results presented in our manuscript.

Prerequisites
Please ensure you have the required environment installed (e.g., PyTorch, OpenCV). You can install the basic dependencies using: pip install -r requirements.txt.

Step-by-Step Instructions

Step 1: Download Pre-trained Model
Before running the code, please download the pre-trained Flowmag model weights.
1. Download the model file from: https://drive.google.com/file/d/1ESSaea-Roe1feFugPFycW5Dd7QCg2ZXR/view
2. Place the downloaded model file (e.g., raft_chkpt_00140.pth) into the checkpoints directory.

Step 2: Convert Video to Image Frames
The algorithm processes image sequences. First, convert your input video into individual frames using the v2p script.
python v2p.py --video_path ./data/input_video.mp4 --save_dir ./data/Bridge/ 
python draw_mask.py --image_path ./data/frames/0001.png --save_path ./data/mask.png 
Finally, run inference_freq.py to generate the magnified video.

---

# Evaluation and Analysis (Post-Processing)
Once all the displacement extraction and motion magnification scripts (for both the baselines and our proposed NDDTCWT method) have been successfully executed, the final quantitative and qualitative analyses are performed using the following provided MATLAB scripts:

Video Quality Analysis: Run the deal_video.m script in MATLAB. This script analyzes all the generated magnified .avi videos to evaluate their visual quality, extract spatial-temporal slices, and compare the perceptual results across different methods.
Displacement Error Computation: Run the errorforall.m script in MATLAB. This script loads the extracted displacement results (e.g., Result_Bridge_Filtered_Data.csv, udt_displacement.csv, and accelerometer data), aligns them, and computes the final quantitative evaluation metrics (e.g., RMSE) to reproduce the tables presented in our manuscript.