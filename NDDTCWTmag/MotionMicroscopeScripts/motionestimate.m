function motionestimate(dataDir, outDir)
    video_file = 'Bridge.avi';
    acceleration_file = 'A2192_datafiles';
    
    vr = VideoReader(fullfile(dataDir, video_file));
    ar = load(fullfile(dataDir, acceleration_file));
    
    % Modal parameters
    amplification = [400, 250];
    loCutoff =[0.1,2.7]%[1.6, 0.1] %[1.6, 2.4];
    hiCutoff =[5, 5]%[2.7, 1.6] %[1.8, 2.7];
    sigma = 3;
    samplingRate = 30;
    temporalFilter = @Butterworth2ndOrder;
    frameRange = [200 400];
    
    % Points to plot the motion in (y, x) format
    % First two are coincident with the accelerometers, last one is not
    pt = [80, 271;
    396 310;
    396 697];
    
    %% Integration and highpass of acceleration to displacement
    % Clip the accleration to a range around when the video was taken, so
    % low -frequency information outside that range doesn't
    % affect integration
    clipRange = 100001:120000;
    acceleration{1} = ar.A2192(clipRange);
    acceleration{2} = ar.A2193(clipRange);
    accelerometerTimestamps = ar.RunTime(clipRange);
    
    % High pass acceleration prior to integration to avoid amplifying
    % low-frequency noise
    accelerationHighpassCutoff = 0.5; % Hz
          

    %% Video processing    
    ylims{1} = [-0.25 0.25];
    ylims{2} = [-0.5 0.5];
    mmPerPixel = -24/42*25.4;
    for mode =  1:2
        
        % Motion amplification
        
%         MotionAmplificationFunction = @(x) phaseAmplify(vr, amplification(mode), ...
%             loCutoff(mode), hiCutoff(mode), samplingRate, outDir, ...
%             'sigma', sigma, 'temporalFilter', temporalFilter, ...
%             'useFrames', frameRange);
%         motion_amp_filename = fullfile(outDir, sprintf('Bridge_amp%d.mat', mode));
%         filename{mode} = RunOnceOnly(motion_amp_filename, MotionAmplificationFunction);

        
        % Create timeslices of original and magnified videos
        
        nyquistRate = samplingRate/2;
        [B_high, A_high] = butter(1, 0.5/nyquistRate, 'high');%阶数 截止频率  类型定义截止频率是高通或低通频率
        [B_band, A_band] = butter(1, [loCutoff(mode) hiCutoff(mode)]/nyquistRate);
        % Construct a function that takes the signal and filters it
        % This function can be passed to computeTemporalFilterReduction to
        % compute the reduction in noise powers    
        temporalFilterHigh = @(x) filter(B_high, A_high, x, [], 3);
        temporalFilterBand = @(x) filter(B_band, A_band, x, [], 3);
        temporalFilterMotion = @(x) temporalFilterBand(temporalFilterHigh(x));
        
        
        % Motion computation
        motion_filename = fullfile(outDir, sprintf('Bridge_mode%d.mat', mode));
        MotionComputationFunction = @(x) computeMotionAndCovarianceudtcwt(vr, ...
            'noiseModel', NoiseModel.GetPtGreyGS323S6MNoiseModel, 'sigma', sigma, ...
             'temporalFilter', temporalFilterMotion, 'useFrames', frameRange);%
        %[motion, motion_covariance] = RunOnceOnly(motion_filename, MotionComputationFunction);
        motion = MotionComputationFunction(motion_filename);
        
        % Camera time is offset from accelerometer
        timeOffset = 88.55 + (13700+frameRange(1))/samplingRate;
        video_time = (timeOffset+1/samplingRate):1/samplingRate:(timeOffset + size(motion,4)/samplingRate);


        
        for ptIDX = 1:3
            % Extract vertical motion at points of interest
            video_displacement = squeeze(motion(pt(ptIDX,1), pt(ptIDX, 2), 2, :));
                 % [B_high, A_high] = butter(1, 0.5/(1/200), 'high');
                 % video_displacement = filter(B_high, A_high, video_displacement);
    
            %[video_displacement, noiseReductionFactor] = FilterDisplacement(video_displacement, ...
            %    accelerationHighpassCutoff, [loCutoff(mode), hiCutoff(mode)], ...
            %    samplingRate);
            
            video_displacement = mmPerPixel * video_displacement;
            %video_error_sd = mmPerPixel * sqrt(motion_covariance(pt(ptIDX,1), pt(ptIDX,2), 2, 2) * 1);
            
            if (ptIDX <3)
                accel_displacement = IntegrateAcceleration(acceleration{ptIDX}, ...
                accelerometerTimestamps, accelerationHighpassCutoff, ...
                [loCutoff(mode), hiCutoff(mode)]);     
            else
               accel_displacement = []; 
            end
            outFilename = fullfile(outDir, sprintf('mode%0.1f-%0.1f_pt%d.pdf', loCutoff(mode), hiCutoff(mode), ptIDX));
            %PlotData(video_displacement, video_time , accel_displacement, accelerometerTimestamps, video_error_sd, ylims{mode}, outFilename);
            PlotData(video_displacement, video_time , accel_displacement, accelerometerTimestamps, ylims{mode}, outFilename);
        end
            
    end    
    
end

function PlotData(video_data, video_time, accel_data, accel_time,  ylims, filename)%error_sd,
   ff = figure('Color', [1 1 1], 'Position', [1 1 800 200], 'PaperSize', [8 2],  'PaperUnits', 'inches', 'PaperPosition', [0 0 8 2]);


   plot(video_time, video_data);
   hold on;
   if (not(isempty(accel_data)))
      plot(accel_time, accel_data, 'r--'); 
   end
   % Plot a horizontal line for error bars
%    plot([0, 1700000], [1, 1]*error_sd, 'm', 'LineWidth', 1);
%    plot([0, 1700000], -[1,1]*error_sd, 'm', 'LineWidth', 1);
   xlabel('Time (s)');
   ylabel('Displacement (mm)');
   ff.Children.Position = [0.25 0.25 0.72 0.72];
   set(gca, 'Xtick', 545+[0 10 20 30]);
   set(gca, 'Ytick', ylims*0.6);
   xlim([545 567]);
   ylim(ylims);
   saveas(ff, filename);
end

% Filters the signal with a highpass filter and then a bandpass filter and 
% computes the reduction in noise piwer
function [filteredSignal, reductionFactor] = FilterDisplacement(signal, highpassCutoff, bandpassCutoff, samplingRate)
    nyquistRate = samplingRate/2;
    [B_high, A_high] = butter(1, highpassCutoff/nyquistRate, 'high');
    [B_band, A_band] = butter(1, bandpassCutoff/nyquistRate);
    % Construct a function that takes the signal and filters it
    % This function can be passed to computeTemporalFilterReduction to
    % compute the reduction in noise powers    
    nF = numel(signal);
    temporalFilterHigh = @(x) filter(B_high, A_high, double(squeeze(x)));
    temporalFilterBand = @(x) filter(B_band, A_band, squeeze(x));
    temporalFilter = @(x) temporalFilterBand(temporalFilterHigh(x));
        
    filteredSignal = temporalFilter(signal);
    reductionFactor = computeTemporalFilterReduction(temporalFilter, nF);
end

function displacementBandpassed = IntegrateAcceleration(acceleration, accelTimestamps, highpassCutoff, bandpassCutoffs)
    accelerometerSamplingRate = 1./(accelTimestamps(2)-accelTimestamps(1));
    accelerometerNyquistRate = accelerometerSamplingRate / 2;
    
    % Units of acceleration coming in are in g (9.86 m/s^2)
    acceleration = acceleration * 9.86 * 1000; % units of millimeters
    
    % High pass acceleration prior to integration to avoid amplifying
    % low-frequency noise
    [B_high, A_high] = butter(1, highpassCutoff/accelerometerNyquistRate, 'high');
    [B_band, A_band] = butter(1, bandpassCutoffs/accelerometerNyquistRate);
    accelerationDetrended = detrend(acceleration);%去除零漂
    % Use zero-phase filtfilt to highpass to avoid causing misalignment
    % issues with video data
    accelerationHighpass = filter(B_high, A_high, accelerationDetrended);
    % Use trapezoidal integration twice
    velocity = cumtrapz(accelTimestamps, accelerationHighpass);
    velocity = detrend(velocity);
    displacement = cumtrapz(accelTimestamps, velocity);
    displacementBandpassed = filter(B_band, A_band, displacement); % m
    
    
end

function displacement = IntegrateAccelerationonline(acceleration, accelTimestamps, highpassCutoff, bandpassCutoffs)
    accelerometerSamplingRate = 1./(accelTimestamps(2)-accelTimestamps(1));
    accelerometerNyquistRate = accelerometerSamplingRate / 2;
    
    % Units of acceleration coming in are in g (9.86 m/s^2)
    acceleration = acceleration * 9.86 * 1000; % units of millimeters
    
    % High pass acceleration prior to integration to avoid amplifying
    % low-frequency noise
    [B_high, A_high] = butter(1, highpassCutoff/accelerometerNyquistRate, 'high');
   % [B_band, A_band] = butter(1, bandpassCutoffs/accelerometerNyquistRate);
    accelerationDetrended = detrend(acceleration);%去除零漂
    % Use zero-phase filtfilt to highpass to avoid causing misalignment
    % issues with video data
    accelerationHighpass = filter(B_high, A_high, accelerationDetrended);
    % Use trapezoidal integration twice
    velocity = cumtrapz(accelTimestamps, accelerationDetrended);
    velocity = detrend(velocity);
    displacement = cumtrapz(accelTimestamps, velocity);
    %displacementBandpassed = filter(B_band, A_band, displacement); % m
    
    
end