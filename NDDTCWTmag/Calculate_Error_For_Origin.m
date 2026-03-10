%% Fig 4 数据生成脚本：含方向统计 (Mean + Std)
clear; clc; close all;
setPath
%% 1. 初始化
si = 128; 
shift_amount = [  10, 0 ]; % 平移 1 像素
im_orig = zeros(si, si); im_orig(si/2, si/2) = 1;
im_shifted = circshift(im_orig, shift_amount);
vidFFT_orig = single(fftshift(fft2(im_orig)));
vidFFT_shifted = single(fftshift(fft2(im_shifted)));

%% 2. 计算 CSP 统计数据 (4个方向)
disp('正在计算 CSP (4 orientations)...');
ht = maxSCFpyrHt(zeros(si,si));
filters = getFilters([si si], 2.^[0:-1:-ht], 4); 
[croppedFilters, filtIDX] = getFilterIDX(filters);
buildLevel = @(im_dft, k) ifft2(ifftshift(croppedFilters{k}.* im_dft(filtIDX{k,1}, filtIDX{k,2})));

csp_stats = zeros(3, 2); % 3层 x [Mean, Std]
current_filter_idx = 2; 

for layer = 1:3
    orient_errors = []; % 收集该层所有方向的误差
    % orient_mae = [];
    orient_std = [];
    fprintf('>>> CSP Layer %d 详细数据:\n', layer);
    for orient = 1:4
        % 理论值 vs 实际值
        resp_orig = buildLevel(vidFFT_orig, current_filter_idx);
  
        target = circshift(resp_orig, shift_amount);
        resp_shifted = buildLevel(vidFFT_shifted, current_filter_idx);
        
        % 计算 RMSE
        diff = abs(resp_shifted - target);
        
        rmse = sqrt(mean(diff(:).^2));
        % mae = mean(diff(:));
        orient_errors(end+1) = rmse;
        orient_std(end+1) = mean(std(diff));
        % orient_mae(end+1) = mae;
        fprintf('    Orient %d: %e\n', orient, mean(abs(diff(:))));
        current_filter_idx = current_filter_idx + 1;
    end
        figure
        surf(abs(resp_orig))
        set(gca, 'ZScale', 'log')
    % 计算统计量
    csp_stats(layer, 1) = mean(orient_errors); % 平均值
    csp_stats(layer, 2) = mean(orient_std);  % 标准差
end
zzpyr = abs(buildLevel(vidFFT_shifted, 5));
resp_orig = buildLevel(vidFFT_orig, 5);
target = circshift(resp_orig, shift_amount);
resp_shifted = buildLevel(vidFFT_shifted, 5);
zzpyrdiss = abs(resp_shifted - target)

%% 3. 计算 NDDTCWT 统计数据 (6个方向)
disp('正在计算 NDDTCWT (6 orientations)...');
[Faf, Fsf] = NDAntonB2; [af, sf] = NDdualfilt1;
J = 3;
pyr_orig = NDxWav2DMEX(im_orig, J, Faf, af, 1);
pyr_shifted = NDxWav2DMEX(im_shifted, J, Faf, af, 1);

nd_stats = zeros(3, 2); % 3层 x [Mean, Std]

for scale = 1:J
    scale_errors = [];
    % scale_mae = [];
    scale_std = [];
    % 遍历所有 2个树 x 3个方向 = 6个方向
    for tree = 1:2
        for group = 1:2
        for orient = 1:3 
            try
                c_orig = pyr_orig{scale}{tree}{group}{orient};
                c_shift = pyr_shifted{scale}{tree}{group}{orient};
                c_target = circshift(c_orig, shift_amount);
                % figure
                % surf(pyr_orig{scale}{tree}{group}{orient})
                % set(gca, 'ZScale', 'log')
                diff = abs(c_shift - c_target);
                % figure
                % surf(diff)
                rmse = sqrt(mean(diff(:).^2));
                
                % mae = mean(diff(:));
                scale_errors(end+1) = rmse;
                scale_std(end+1) = std(diff);
                % scale_mae(end+1) = mae;
                % fprintf('    Tree %d, Group %d, Dir %d: %e\n', tree, group, orient, mean(abs(diff(:))));
                count = count + 1;
            catch
            end
        end
        end
    end
    
    % NDDTCWT 误差极小，标准差也极小，但也算出来
    % 强制把 0 替换为 1e-16 以便画对数图
    mean_val = mean(scale_errors);
    if mean_val < 1e-16, mean_val = 1e-16; end
    
    nd_stats(scale, 1) = mean_val; 
    nd_stats(scale, 2) = mean(scale_std); 
end

min_positive = min(pyr_orig{2}{1}{1}{2}(pyr_orig{2}{1}{1}{2} > 0));
    Z_fixed = pyr_orig{2}{1}{1}{2}; 
    Z_fixed(Z_fixed <= 0) = min_positive;
    figure
    surf(Z_fixed)
    set(gca, 'ZScale', 'log')
    % figure
    % set(gca, 'ZScale', 'log')
    % surf(pyr_shifted{2}{1}{1}{2})

%% 4. 输出 Origin 数据表

disp('==========================================================');
disp('【请复制以下数据块到 Origin】');
disp('----------------------------------------------------------');
disp('Layer   CSP_Mean      CSP_SD        ND_Mean       ND_SD');
disp('----------------------------------------------------------');
for i = 1:3
    fprintf('%d       %e  %e  %e  %e\n', ...
        i, csp_stats(i,1), csp_stats(i,2), nd_stats(i,1), nd_stats(i,2));
end
disp('==========================================================');