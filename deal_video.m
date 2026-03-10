clc; clear; close all;
%% === 1. Settings ===
% 1.1 File Path Settings
dist_file = 'masked_x200_f1-998_bp_1.600-1.800Hz.mp4'; % Test video
ref_file = 'Bridge.avi';                                          % Reference video
save_path = 'bridge\';                             % Result saving path
if ~exist(save_path, 'dir'), mkdir(save_path); end

% 1.2 Y-T Spatiotemporal Slice Parameters (Supports specified frame range)
target_x = 271;                 % Specified X coordinate for extraction
y_range = 40:140;              % Specified Y coordinate range for extraction
slice_start_frame = 1;          % Y-T slice start frame
slice_end_frame = 400;         % Y-T slice end frame

% 1.3 Parameters for extracting and saving a specific full frame
frame_to_extract = 100;          % Specify which frame of the test video to save as a full image

% 1.4 PSNR / SSIM Calculation Parameters (Supports specified frame range mapping)
REF_START_FRAME = 1;         % Reference video start frame
REF_END_FRAME   = 400;         % Reference video end frame 
DIST_START_FRAME = 1;           % Test video start frame

%% === 2. Extract and Save Specific Single Full Frame ===
disp(['Extracting frame ', num2str(frame_to_extract), ' from the test video...']);
v_dist = VideoReader(dist_file);
frame_data = read(v_dist, frame_to_extract); 
imwrite(frame_data, fullfile(save_path, ['Full_Frame_', num2str(frame_to_extract), '.png']));

%% === 3. Generate Y-T Spatiotemporal Slice Image (With Axes Version) ===
disp(['Generating Y-T spatiotemporal slice (from frame ', num2str(slice_start_frame), ' to ', num2str(slice_end_frame), ')...']);
num_slice_frames = slice_end_frame - slice_start_frame + 1;
height_slice = length(y_range);
slice_img = zeros(height_slice, num_slice_frames, 3, 'uint8');

for k = 1:num_slice_frames
    current_idx = slice_start_frame + k - 1;
    temp_frame = read(v_dist, current_idx);
    slice_img(:, k, :) = temp_frame(y_range, target_x, :);
end
%slice_img = fliplr(slice_img); % Flip the image matrix left-to-right
imwrite(slice_img, fullfile(save_path, ['Raw_YT_Slice_x', num2str(target_x), '.png']));

disp('Adding axes and scale bar to the slice, and exporting high-resolution image...');
[img_height, img_width, ~] = size(slice_img);
figure('Color', 'w', 'Position', [100, 100, 800, 300]); 
imshow(slice_img, 'XData', [1, img_width], 'YData', [1, img_height]);
axis on; hold on;
axis tight; 

% Set axis text
xlabel('Time (Frames)', 'FontSize', 14, 'FontName', 'Arial', 'FontWeight', 'bold');
ylabel('Position (pixels)', 'FontSize', 14, 'FontName', 'Arial', 'FontWeight', 'bold');
set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'FontName', 'Arial');
set(gca, 'XTick', [0, 100, 200, 300, 400]); 
set(gca, 'Units', 'normalized', 'Position', [0.12, 0.25, 0.82, 0.65]);

% -----------------------------------------------------------
% Add internal scale bar - Retain magnification in axes version for layout formatting
% -----------------------------------------------------------
scale_len_px = 20; % Scale bar represents 20 pixels
bar_width = 3;     % Line thickness
margin_x = 20;     % Right margin
margin_y = 10;     % Bottom margin

pos_x = img_width - margin_x; 
pos_y_start = img_height - margin_y - scale_len_px;
pos_y_end = img_height - margin_y;

plot([pos_x, pos_x], [pos_y_start, pos_y_end], 'Color', 'w', 'LineWidth', bar_width);

text(pos_x - 5, pos_y_start + scale_len_px/2, '20 px', ...
    'Color', 'w', ...
    'FontSize', 12, ...
    'FontName', 'Arial', ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right');

exportgraphics(gcf, fullfile(save_path, ['HighRes_YT_Slice_x', num2str(target_x), '.tif']), 'Resolution', 300);

%% === 3.5 Generate an Extra Pure Version (Only image + scale bar, absolutely no enlarging of the original image) ===
disp('Exporting extra pure scale bar image without axes...');

% Create an invisible canvas
f_pure = figure('Visible', 'off', 'Color', 'w'); 

% Display the image, use 'Border', 'tight' to ensure absolutely no borders
imshow(slice_img, 'Border', 'tight');
hold on;

% --- Draw pure label using the requested format ---
margin = 5; 
pos_x_pure = img_width - margin;
pos_y_start_pure = img_height - margin - scale_len_px;
pos_y_end_pure = img_height - margin;

% Draw line (Line width set to 2)
plot([pos_x_pure, pos_x_pure], [pos_y_start_pure, pos_y_end_pure], 'Color', 'w', 'LineWidth', 4);

% Add text "20 px" (Offset set to -2 for compact layout)
text(pos_x_pure - 2, pos_y_start_pure + scale_len_px/2, '20 px', ...
    'Color', 'w', 'FontSize', 24, 'FontWeight', 'bold', ...
    'FontName', 'Arial', 'HorizontalAlignment', 'right');

% --- Export ---
pure_filename = fullfile(save_path, ['Pure_ScaleBar_YT_Slice_x', num2str(target_x), '.png']);
% [Key Point]: Do not use '-r300' to forcibly enlarge, use default resolution to export compactly, maintaining 1:1 pixel size
exportgraphics(gca, pure_filename);
close(f_pure); % Close canvas

disp(['Pure version saved (Not enlarged): ', pure_filename]);

%% === 4. Calculate Metrics (PSNR / SSIM / RMSE) ===
num_frames_to_process = REF_END_FRAME - REF_START_FRAME + 1;

try
    v_ref = VideoReader(ref_file);
catch ME
    error(['Unable to read the reference video file, please check the filename or path! Error message: ' ME.message]);
end

fprintf('\n--- Video Information ---\n');
fprintf('Reference video (%s) total frames: %d\n', ref_file, v_ref.NumFrames);
fprintf('Test video (%s) total frames: %d\n', dist_file, v_dist.NumFrames);
fprintf('------------------\n');

if v_ref.NumFrames < REF_END_FRAME
    error('Error: The total number of frames in the reference video is less than the set end frame.');
end
if v_dist.NumFrames < DIST_START_FRAME + num_frames_to_process - 1
    error('Error: The total number of frames in the test video is insufficient to cover the required comparison range.');
end

target_width = v_dist.Width;
target_height = v_dist.Height;

disp(['Target resolution set to test video size: ' num2str(target_width) 'x' num2str(target_height)]);

psnr_list = zeros(num_frames_to_process, 1);
ssim_list = zeros(num_frames_to_process, 1);
rmse_list = zeros(num_frames_to_process, 1);
frame_counter = 0; 

for i = 1 : num_frames_to_process
    frame_counter = frame_counter + 1; 
    
    dist_idx = DIST_START_FRAME + i - 1;
    ref_idx  = REF_START_FRAME + i - 1;
    
    img_ref_rgb = read(v_ref, ref_idx);
    img_dist_rgb = read(v_dist, dist_idx);
    
    % Resize reference frame to match test frame
    if size(img_ref_rgb, 1) ~= target_height || size(img_ref_rgb, 2) ~= target_width
        img_ref_rgb = imresize(img_ref_rgb, [target_height, target_width], 'bilinear');
    end
    
    % Convert to grayscale image
    img_ref = rgb2gray(img_ref_rgb);
    img_dist = rgb2gray(img_dist_rgb);
    
    if isequal(img_ref, img_dist)
        psnr_val = Inf; 
        ssim_val = 1.0; 
        rmse_val = 0.0; 
    else
        psnr_val = psnr(img_dist, img_ref);
        ssim_val = ssim(img_dist, img_ref);
        img_ref_double = double(img_ref);
        img_dist_double = double(img_dist);
        mse = mean((img_ref_double(:) - img_dist_double(:)).^2);
        rmse_val = sqrt(mse);
    end
    
    psnr_list(frame_counter) = psnr_val;
    ssim_list(frame_counter) = ssim_val;
    rmse_list(frame_counter) = rmse_val;
  
end

avg_psnr_temp = mean(psnr_list);
if isinf(avg_psnr_temp)
    avg_psnr_str = 'Inf';
else
    avg_psnr_str = sprintf('%.4f', avg_psnr_temp);
end

fprintf('\n================ Final Results ================\n');
fprintf('Reference video: %s\n', ref_file);
fprintf('Test video: %s\n', dist_file);
fprintf('Average PSNR: %s dB\n', avg_psnr_str);
fprintf('Average SSIM: %.4f\n', mean(ssim_list));
fprintf('Average RMSE: %.4f\n', mean(rmse_list));
fprintf('========================================\n');

% Save CSV file to the specified directory
Frame_Index = (DIST_START_FRAME:(DIST_START_FRAME + frame_counter - 1))'; 
T = table(Frame_Index, psnr_list, ssim_list, rmse_list, 'VariableNames', {'Dist_Frame_Index', 'PSNR', 'SSIM', 'RMSE_Image'});
csv_filename = fullfile(save_path, ['Metrics_Mapped_', num2str(REF_START_FRAME), '_', num2str(REF_END_FRAME), '.csv']);
writetable(T, csv_filename);

disp(['Detailed frame-by-frame data has been saved as a CSV file: ', csv_filename]);
disp('All tasks completed successfully!');