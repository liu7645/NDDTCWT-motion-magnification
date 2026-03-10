clc; clear; close all;

%% 1. Read CSV Data Individually
fprintf('Reading data...\n');

% --- 1. Ground Truth ---
% Filename: accel_displacement.csv
T_accel = readtable('NDDTCWTmag/output/accel_displacement.csv');
t_gt = T_accel.time;
y_gt = T_accel.acceleration; 

% --- 2. Method 1: Phase-based (PYR) ---
% Filename: pyr_displacement.csv
T_pyr = readtable('NDDTCWTmag/output/pyr_displacement.csv');
t_pyr = T_pyr.videotime;
y_pyr = T_pyr.pyr;

% --- 3. Method 2: NDDTCWT ---
% Filename: udt_displacement.csv
T_udt = readtable('NDDTCWTmag/output/udt_displacement.csv');
t_udt = T_udt.videotime;
y_udt = T_udt.pyr; % Note: Header remains 'pyr' in this CSV

% --- 4 & 5. Method 3 & 4: RAFT & DIS ---
% Filename: Result_Bridge_Filtered_Data.csv
T_rd = readtable('RAFTDIS/Result_Bridge_Filtered_Data.csv');

% RAFT
t_raft = T_rd.Time_Video;
y_raft = T_rd.RAFT_P1_Y_Filtered;

% DIS
t_dis = T_rd.Time_Video;
y_dis = T_rd.DIS_P1_Y_Filtered;

%% 2. Core Step: Interpolation and Alignment
% Logic: Use the time axis of each method to query corresponding ground truth values
fprintf('Aligning data and calculating errors...\n');

% Align PYR
gt_at_pyr = interp1(t_gt, y_gt, t_pyr, 'pchip');
err_pyr = abs(y_pyr - gt_at_pyr);

% Align UDT
gt_at_udt = interp1(t_gt, y_gt, t_udt, 'pchip');
err_udt = abs(y_udt - gt_at_udt);

% Align RAFT
gt_at_raft = interp1(t_gt, y_gt, t_raft, 'pchip');
err_raft = abs(y_raft - gt_at_raft);

% Align DIS
gt_at_dis = interp1(t_gt, y_gt, t_dis, 'pchip');
err_dis = abs(y_dis - gt_at_dis);

%% 3. Calculate RMSE for Summary
% Ignore NaNs resulting from non-overlapping time boundaries
rmse_phase = sqrt(nanmean(err_pyr.^2));
rmse_nd    = sqrt(nanmean(err_udt.^2));
rmse_raft  = sqrt(nanmean(err_raft.^2));
rmse_dis   = sqrt(nanmean(err_dis.^2));

fprintf('\nRMSE Results:\n');
fprintf('Phase-based: %.5f\n', rmse_phase);
fprintf('NDDTCWT:     %.5f\n', rmse_nd);
fprintf('RAFT:        %.5f\n', rmse_raft);
fprintf('DIS:         %.5f\n', rmse_dis);

%% 4. Figure 1: Displacement Time History Comparison
fprintf('Plotting displacement comparison...\n');
figure('Name', 'Displacement Comparison', 'Position', [100, 100, 900, 400]);
hold on; grid on;

plot(t_gt, y_gt, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Ground Truth (Accel)');
plot(t_pyr, y_pyr, 'b--', 'LineWidth', 1.2, 'DisplayName', 'Phase-based');
plot(t_udt, y_udt, 'r-.', 'LineWidth', 1.2, 'DisplayName', 'NDDTCWT');
plot(t_raft, y_raft, 'g:', 'LineWidth', 1.5, 'DisplayName', 'RAFT');
plot(t_dis, y_dis, 'm-', 'LineWidth', 1, 'DisplayName', 'DIS');

% Set chart attributes
xlabel('Time (s)', 'FontSize', 12);
ylabel('Displacement', 'FontSize', 12);
title('Displacement vs. Time Comparison', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);

% Limit X-axis to the common time range of all vision methods
common_min = max([min(t_pyr), min(t_udt), min(t_raft)]);
common_max = min([max(t_pyr), max(t_udt), max(t_raft)]);
xlim([common_min common_max]);

hold off;

%% 5. Figure 2: Error Comparison (Absolute Error vs. GT)
fprintf('Plotting error comparison...\n');
figure('Name', 'Error Comparison', 'Position', [150, 150, 900, 400]);
hold on; grid on;

plot(t_pyr, err_pyr, 'b-', 'LineWidth', 1.2, 'DisplayName', 'Error Phase-based');
plot(t_udt, err_udt, 'r-', 'LineWidth', 1.2, 'DisplayName', 'Error NDDTCWT');
plot(t_raft, err_raft, 'g-', 'LineWidth', 1.2, 'DisplayName', 'Error RAFT');
plot(t_dis, err_dis, 'm-', 'LineWidth', 1.2, 'DisplayName', 'Error DIS');

% Set chart attributes
xlabel('Time (s)', 'FontSize', 12);
ylabel('Absolute Error', 'FontSize', 12);
title('Absolute Error Relative to Ground Truth', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
xlim([common_min common_max]);

hold off;
fprintf('Process completed successfully!\n');
%% 6. Save Errors to Excel Sheets
fprintf('Saving error data to Excel...\n');

filename = 'Method_Errors_Results.xlsx';
writetable(table(t_pyr, err_pyr),   filename, 'Sheet', 'Phase-based');
writetable(table(t_udt, err_udt),   filename, 'Sheet', 'NDDTCWT');
writetable(table(t_raft, err_raft), filename, 'Sheet', 'RAFT');
writetable(table(t_dis, err_dis),   filename, 'Sheet', 'DIS');

fprintf('Error data saved to: %s\n', filename);