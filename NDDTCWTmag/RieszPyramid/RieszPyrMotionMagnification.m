% OUTNAME = RIESZPYRMOTIONMAGNIFICATION(VR, AMPLIFICATION_FACTOR, LOW_CUTOFF, HIGH_CUTOFF, VIDEO_SAMPLING_RATE, OUTDIR, VARARGIN)
%     
% Computes a motion magnified video, amplifying motions in the specified
% temporal band. Uses the Riesz pyramid method, which is typically faster
% than the phase-based method and almost the same quality. Returns the
% filename of the output motion magnified video. Motion magnifies the motions
% that are within a passband of LOW_CUTOFF to HIGH_CUTOFF Hz by MAGPHASE times. 
% VIDEO_SAMPLING_RATE can be different than the frame rate specified in VR, 
% especially if VR is a high-speed video. OUTDIR is the output directory. 
% If temporalFilter (see below) is not 'online', the passband parameters
% may not be used.
% 
% Required arugments:
% vr
%   - A class that has properties Width, Height and NumberOfFrames and 
%     implements the function read, which takes either an integer or a 
%     tuple of integers and returns a HeightxWidthxChannels array in the
%     first case and a HeightxWidthxChannelsxFrames array in the second
%     case corresponding to the number of frames between the tuple of
%     integers. Should be of integer type of either 8 bit or 16 bit.
%     Example classes are the in-built VideoReader and the included
%     CineReader, MyVideoReader and FileReader
%
% Optional arguments:
% buildPyr, reconPyr        (buildLpyrFast, reconLpyrFast)
%   - Which pyramid to use as the real part of the Riesz pyramid.
%     Both arguments should be specified together and should be inverse
%     transforms of each other.
% scaleVideo                (1, no scaling)
%   - Scale frames of video before any processing
% sigma                     (3 px)            
%   - Amount of spatial smoothing (in px)
% temporalFilter            'online' 
%   - What temporal filter to use. If temporalFilter is online, a first order
%     butterworth filter will be used that processes the frames as they come.
%     Otherwise, temporalFilter should be a function that process the third 
%     dimension of an input array and takes three argument. Specifiying a 
%     temporal filter will load all phases into memory resulting in slower 
%     performance.  LOW_CUTOFF/VIDEO_SAMPLING_RATE and
%     HIGH_CUTOFF/VIDEO_SAMPLING_RATE are passed as the 2nd and 3rd
%     argument to temporalFilter, but they do not need to be used.
% useFrames                 ([1, vr.NumberOfFrames])
%   - A tuple specifying a subset of frames to process
%
% Author(s): Neal Wadhwa
% License: Please refer to LICENSE file
% Date: August 2016
function outName = RieszPyrMotionMagnification(vr, amplification_factor, low_cutoff, high_cutoff, video_sampling_rate, outDir, varargin)
    
    p = inputParser();
    
    defaultSigma = 2;
    defaultScale = 1;
    defaultFrames = [1, vr.NumberOfFrames];
    defaultTemporalFilter = 'online';
    defaultBuildPyr = @buildLpyrFast;
    defaultReconPyr = @reconLpyrFast;
    
    addOptional(p, 'sigma', defaultSigma, @isnumeric);   
    addOptional(p, 'temporalFilter', defaultTemporalFilter);
    addOptional(p, 'scaleVideo', defaultScale);
    addOptional(p, 'useFrames', defaultFrames);
    addOptional(p, 'buildPyr', defaultBuildPyr);    
    addOptional(p, 'reconPyr', defaultReconPyr);
    
    parse(p, varargin{:});
    
    sigma              = p.Results.sigma;
    temporalFilter     = p.Results.temporalFilter;
    scale_factor       = p.Results.scaleVideo;
    frameRange         = p.Results.useFrames;
    buildPyr           = p.Results.buildPyr;
    reconPyr           = p.Results.reconPyr;

    
    readFrame = @(k) imresize(im2single(vr.read(frameRange(1)+k-1)), scale_factor);
    nF = frameRange(2)-frameRange(1)+1;    

    
    if (and(isstr(temporalFilter), strcmp(temporalFilter, 'online')))
       % Use online first order butterworth filter 
       temporalFilterStr = 'online';
       onlineFilter = true;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       % Initializes spatial smoothing kernel and temporal filtering
       % coefficients.

       % Compute an IIR temporal filter coefficients. Butter could be replaced
       % with any IIR temporal filter. Lower temporal_filter_order is faster
       % and uses less memory, but is less accurate. See pages 493-532 of
       % Oppenheim 3rd ed.    
       nyquist_frequency = video_sampling_rate/2;
       temporal_filter_order = 1; 
       [B, A] = butter(temporal_filter_order, [low_cutoff, high_cutoff]/nyquist_frequency);
    else 
        % Precompute phases and filter prior to amplification
        % temporalFilter should be a function that takes three arguments
        % and filters along the third dimension passed into it.
        temporalFilterStr = func2str(temporalFilter);
        onlineFilter = false;
    end
    
    outName = fullfile(outDir, sprintf('%s-%s-band%0.2f-%0.2f-sr%d-alpha%d-sigma%d-scale%0.2f-frames%d-%d-%s-RieszPyr.avi', ...
         vr.Name, temporalFilterStr, low_cutoff, high_cutoff, video_sampling_rate, ...
         amplification_factor, sigma, scale_factor, frameRange(1), frameRange(2), func2str(buildPyr)));
       vw = VideoWriter(outName);
       vw.FrameRate = vr.FrameRate;
       vw.Quality = 96;
       vw.open();
       
    
    
    % Computes convolution kernel for spatial blurring kernel used during
    % phase denoisng step.
    gaussian_kernel_sd = 2; % px
    gaussian_kernel = fspecial('gaussian', 4*ceil(gaussian_kernel_sd)+1, gaussian_kernel_sd);

    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialization of variables before main loop.
    previous_frame = rgb2y(readFrame(1));
    [previous_laplacian_pyramid, previous_riesz_x, previous_riesz_y] = ComputeRieszPyramid(previous_frame, buildPyr);  
    number_of_levels = numel(previous_laplacian_pyramid) - 1; % Do not include lowpass residual            
    for l = 1:number_of_levels
       % Initializes current value of quaternionic phase. Each coefficient
       % has a two element quaternionic phase that is defined as 
       % phase times (cos(orientation), sin(orientation))
       % It is initialized at zero
       phase_cos{l} = zeros(size(previous_laplacian_pyramid{l}));
       phase_sin{l} = zeros(size(previous_laplacian_pyramid{l}));
        
        
       % Initializes IIR temporal filter values. These values are used during 
       % temporal filtering. See the function IIRTemporalFilter for more 
       % details. The initialization is a zero motion boundary condition 
       % at the beginning of the video.    
       register0_cos{l} = zeros(size(previous_laplacian_pyramid{l}));
       register1_cos{l} = zeros(size(previous_laplacian_pyramid{l}));
            
       register0_sin{l} = zeros(size(previous_laplacian_pyramid{l}));            
       register1_sin{l} = zeros(size(previous_laplacian_pyramid{l}));                              
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Main loop. 
    if (onlineFilter)
        for frameIDX = 1:nF
            fprintf('Processing frame %d of %d.\n', frameIDX, nF);
            current_frame = rgb2y(readFrame(frameIDX));
            [current_laplacian_pyramid, current_riesz_x, current_riesz_y] = ComputeRieszPyramid(current_frame, buildPyr);

            % We compute a motion_magnified_laplacian_pyramid first and then
            % collapse it at the end.             
            % The processing in the following loop is processed on each level
            % of the Riesz pyramid independently
            for l = 1:number_of_levels            

                % Compute phase difference between current Riesz pyramid
                % coefficients and previous Riesz pyramid coefficients.       
                [phase_difference_cos, phase_difference_sin, amplitude] = ComputePhaseDifferenceAndAmplitude(current_laplacian_pyramid{l}, current_riesz_x{l}, current_riesz_y{l}, previous_laplacian_pyramid{l}, previous_riesz_x{l}, previous_riesz_y{l});                    


                % Adds the phase difference to the current value of the phase.
                % Computing the current value of the phase in this way is 
                % equivalent to phase unwrapping.
                phase_cos{l} = phase_cos{l} + phase_difference_cos;
                phase_sin{l} = phase_sin{l} + phase_difference_sin;


                % Temporally filter the phase using current value and stored
                % information
                [phase_filtered_cos, register0_cos{l}, register1_cos{l}] = IIRTemporalFilter(B, A, phase_cos{l}, register0_cos{l}, register1_cos{l});
                [phase_filtered_sin, register0_sin{l}, register1_sin{l}] = IIRTemporalFilter(B, A, phase_sin{l}, register0_sin{l}, register1_sin{l});


                % Spatial blur the temporally filtered phase signals
                % This is not an optional step. In addition to denoising,
                % it smooths out errors made during the various approximations.
                phase_filtered_cos = AmplitudeWeightedBlur(phase_filtered_cos, amplitude, gaussian_kernel);
                phase_filtered_sin = AmplitudeWeightedBlur(phase_filtered_sin, amplitude, gaussian_kernel);


                % The motion magnified pyramid is computed by phase shifting
                % the input pyramid by the spatio-temporally filtered phase and
                % taking the real part.
                motion_magnified_laplacian_pyramid{l} = PhaseShiftCoefficientRealPart(current_laplacian_pyramid{l}, current_riesz_x{l}, current_riesz_y{l}, amplification_factor * phase_filtered_cos, amplification_factor * phase_filtered_sin);                          
            end


            % Take lowpass residual from current frame's lowpass residual
            % and collapse pyramid.
            motion_magnified_laplacian_pyramid{number_of_levels+1} = current_laplacian_pyramid{number_of_levels+1};               
            motion_magnified_frames_gray = reconPyr(motion_magnified_laplacian_pyramid);        
            current_frame = rgb2ntsc(readFrame(frameIDX));
            current_frame(:,:,1) = motion_magnified_frames_gray;
            motion_magnified_frame = ntsc2rgb(current_frame);

            % Write or display the motion magnified frame.
            vw.writeVideo(im2uint8(motion_magnified_frame));

            % Prepare for next iteration of loop
            previous_laplacian_pyramid = current_laplacian_pyramid;
            previous_riesz_x = current_riesz_x;
            previous_riesz_y = current_riesz_y;

        end        
    else
        
        % Initialize phases and amplitude storage. We compute all of them 
        % for every frame and then temporally and spatially filter, before
        % using them to motion magnify the frames of the original video.
        for l = 1:number_of_levels
            im_sz = size(previous_laplacian_pyramid{l});
            phase_cos{l} = zeros(im_sz(1), im_sz(2), nF, 'single');
            phase_sin{l} = zeros(im_sz(1), im_sz(2), nF, 'single');
            amplitude{l} = zeros(im_sz(1), im_sz(2), nF, 'single');            
        end
        
        for frameIDX = 2:nF
            fprintf('Computing phase in frame %d of %d.\n', frameIDX, nF);
            current_frame = rgb2y(readFrame(frameIDX));
            [current_laplacian_pyramid, current_riesz_x, current_riesz_y] = ComputeRieszPyramid(current_frame, buildPyr);

            % We compute a motion_magnified_laplacian_pyramid first and then
            % collapse it at the end.             
            % The processing in the following loop is processed on each level
            % of the Riesz pyramid independently
            for l = 1:number_of_levels            

                % Compute phase difference between current Riesz pyramid
                % coefficients and previous Riesz pyramid coefficients.       
                [phase_difference_cos, phase_difference_sin, amp] = ComputePhaseDifferenceAndAmplitude(...
                    current_laplacian_pyramid{l}, current_riesz_x{l}, current_riesz_y{l}, ...
                    previous_laplacian_pyramid{l}, previous_riesz_x{l}, previous_riesz_y{l});                    


                % Adds the phase difference to the current value of the phase.
                % Computing the current value of the phase in this way is 
                % equivalent to phase unwrapping.
                phase_cos{l}(:,:,frameIDX) = phase_cos{l}(:,:,frameIDX-1) + phase_difference_cos;
                phase_sin{l}(:,:,frameIDX) = phase_sin{l}(:,:,frameIDX-1) + phase_difference_sin;
                amplitude{l}(:,:,frameIDX) = amp;
                if (frameIDX == 2)
                   amplitude{l}(:,:,1) = amplitude{l}(:,:,2); 
                end
            end    
            previous_laplacian_pyramid = current_laplacian_pyramid;
            previous_riesz_x = current_riesz_x;
            previous_riesz_y = current_riesz_y;
        end
        
        
        for l = 1:number_of_levels
           fprintf('Spatiotemporally filtering level %d of %d.\n', l, number_of_levels);
           phase_cos{l} = temporalFilter(phase_cos{l}, low_cutoff/video_sampling_rate, high_cutoff/video_sampling_rate); 
           phase_sin{l} = temporalFilter(phase_sin{l}, low_cutoff/video_sampling_rate, high_cutoff/video_sampling_rate);
           phase_cos{l} = AmplitudeWeightedBlur(phase_cos{l}, amplitude{l}, gaussian_kernel);         
           phase_sin{l} = AmplitudeWeightedBlur(phase_sin{l}, amplitude{l}, gaussian_kernel);      
        end
        for frameIDX = 1:nF
          fprintf('Motion magnifying frame %d of %d.\n', frameIDX, nF);
          current_frame = rgb2y(readFrame(frameIDX));
          [current_laplacian_pyramid, current_riesz_x, current_riesz_y] = ComputeRieszPyramid(current_frame, buildPyr);
          motion_magnified_laplacian_pyramid{number_of_levels+1} = current_laplacian_pyramid{number_of_levels+1};
           for l = 1:number_of_levels
               motion_magnified_laplacian_pyramid{l} = PhaseShiftCoefficientRealPart(current_laplacian_pyramid{l}, ...
                   current_riesz_x{l}, current_riesz_y{l}, amplification_factor * phase_cos{l}(:,:,frameIDX), ...
                   amplification_factor * phase_sin{l}(:,:,frameIDX)); 
           end
           motion_magnified_frames_gray = reconPyr(motion_magnified_laplacian_pyramid);        
            current_frame = rgb2ntsc(readFrame(frameIDX));
            current_frame(:,:,1) = motion_magnified_frames_gray;
            motion_magnified_frame = ntsc2rgb(current_frame);            
            vw.writeVideo(im2uint8(motion_magnified_frame));
        end
    end
    
    vw.close();
end


% Compute Riesz pyramid of two dimensional frame. This is done by first 
% computing the laplacian pyramid of the frame and then computing the 
% approximate Riesz transform of each level that is not the lowpass
% residual. The result is stored as an array of grayscale frames.
% Corresponding locations in the result correspond to the real, 
% i and j components of Riesz pyramid coefficients.
function [laplacian_pyramid, riesz_x, riesz_y] = ComputeRieszPyramid(grayscale_frame, ComputeLaplacianPyramid)    
    laplacian_pyramid = ComputeLaplacianPyramid(grayscale_frame);    
    number_of_levels = numel(laplacian_pyramid)-1; 
    
       
    % The approximate Riesz transform of each level that is not the
    % low pass residual is computed. For more details on the approximation,
    % see supplemental material.
    kernel_x = [0.0  0.0  0.0;
                0.5  0.0 -0.5;
                0.0  0.0  0.0];
    kernel_y = [0.0  0.5  0.0;
                0.0  0.0  0.0;
                0.0 -0.5  0.0];
    for l = 1:number_of_levels
        riesz_x{l} = imfilter(laplacian_pyramid{l}, kernel_x);
        riesz_x{l}(:,2) = riesz_x{l}(:,1);
        riesz_x{l}(:,end-1) = riesz_x{l}(:,end);
        
        riesz_y{l} = imfilter(laplacian_pyramid{l}, kernel_y);
        riesz_y{l}(2,:) = riesz_y{l}(1,:);
        riesz_y{l}(end-1,:) = riesz_y{l}(end,:);    
    end
end


% Computes quaternionic phase difference between current frame and previous 
% frame. This is done by dividing the coefficients of the current frame
% and the previous frame and then taking the quaternionic logarithm. We
% assume the orientation at a point is roughly constant to simplify
% the calcuation.
function [phase_difference_cos, phase_difference_sin, amplitude] = ComputePhaseDifferenceAndAmplitude(current_real, current_x, current_y, previous_real, previous_x, previous_y)
    % q_current = current_real + i * current_x + j * current_y
    % q_previous = previous_real + i * previous_x + j * previous_y
    % We want to compute phase difference, which is phase of 
    %    q_current/q_previous
    % This is equal to (See Eq. 10 of tech. report)
    %    q_current * conjugate(q_previous)/||q_previous||^2
    % Phase is invariant to scalar multiples, so we want phase of
    %    q_current * conjugate(q_previous)
    % which we compute now (Eq. 7 of tech. report). We simplify the 
    % processing by assuming the fourth component is zero. This is 
    % roughly true if the orienation at the point is roughly constant
    % between frames.
    q_conj_prod_real = current_real.*previous_real + current_x.*previous_x + current_y.*previous_y;
    q_conj_prod_x = -current_real.*previous_x + previous_real.*current_x;
    q_conj_prod_y = -current_real.*previous_y + previous_real.*current_y;
    
    % Now we take the quaternion logarithm of this (Eq. 12 in tech. report)
    % Only the imaginary part corresponds to quaternionic phase.
    q_conj_prod_amplitude = sqrt(q_conj_prod_real.^2 + q_conj_prod_x.^2 + q_conj_prod_y.^2);
    phase_difference = acos(q_conj_prod_real./(eps+q_conj_prod_amplitude));
    cos_orientation = q_conj_prod_x ./(eps+sqrt(q_conj_prod_x.^2+q_conj_prod_y.^2));
    sin_orientation = q_conj_prod_y ./(eps+sqrt(q_conj_prod_x.^2+q_conj_prod_y.^2));
    
    % This is the quaternionic phase (Eq. 2 in tech. report)    
    phase_difference_cos = phase_difference .* cos_orientation;
    phase_difference_sin = phase_difference .* sin_orientation;
    
    % Under the assumption that changes are small between frames, we can 
    % assume that the amplitude of both coefficients is the same. So,
    % to compute the amplitude of one coefficient, we just take the sqrt
    % of their conjugate product
    amplitude = sqrt(q_conj_prod_amplitude);
end


% Temporally filters phase with IIR filter with coefficients B, A. 
% Given current phase value and value of previously computed registers,
% comptues current temporally filtered phase value and updates registers.
% Assumes filter given by B, A is first order IIR filter, so that 
% B and A have 3 coefficients each. Also, assumes A(1) = 1. Computation
% is Direct Form Type II (See pages 388-390 of Oppenheim 3rd Ed.)
function [temporally_filtered_phase, register0, register1] = IIRTemporalFilter(B, A, phase, register0, register1)
    temporally_filtered_phase = B(1) * phase + register0;
    register0 = B(2) * phase + register1 - A(2) * temporally_filtered_phase;
    register1 = B(3) * phase             - A(3) * temporally_filtered_phase;
end

% Amplitude weighted blur
function spatially_smooth_temporally_filtered_phase = AmplitudeWeightedBlur(temporally_filtered_phase, amplitude, blur_kernel)
    denominator = eps + imfilter(amplitude, blur_kernel);
    numerator   = imfilter(temporally_filtered_phase.*amplitude, blur_kernel);
    spatially_smooth_temporally_filtered_phase = numerator./denominator;
end

% Phase shifts a Riesz pyramid coefficient and returns the real part of the 
% resulting coefficient. The input coefficient is a three
% element quaternion. The phase is two element imaginary quaternion. 
% The phase is exponentiated and then the result is mutiplied by the first
% coefficient. All operations are defined on quaternions.
function result = PhaseShiftCoefficientRealPart(riesz_real, riesz_x, riesz_y, phase_cos, phase_sin)    
    % Quaternion Exponentiation
    phase_magnitude = eps+sqrt(phase_cos.^2+phase_sin.^2);  % \|v\| in Eq. 11 in tech. report.
    exp_phase_real = cos(phase_magnitude);
    exp_phase_x = phase_cos./phase_magnitude.*sin(phase_magnitude);
    exp_phase_y = phase_sin./phase_magnitude.*sin(phase_magnitude);
    
    % Quaternion Multiplication (just real part)
    result = exp_phase_real.*riesz_real ...
             - exp_phase_x.*riesz_x ...
             - exp_phase_y.*riesz_y;
end
