%% 
% [MOTION, MOTION_COVARIANCE] = COMPUTEMOTIONANDCOVARIANCE( VR, VARARGIN )
% 
% Computes MOTION between frames and the first frame. Also computes the
% uncertainity in this estimate arising from camera sensor noise. MOTION
% returns a tuple for every processed tuple. MOTION_COVARIANCE is a 2x2
% matrix for every motion estimate. If nargout == 1 (motion_covariance is
% not specified), then the motion_covariance is not computed.
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
% computeCovarianceForEveryFrame (false)
%   - Compute motion covariance for every frame (memory-intensive and slow)
% covarianceSamples         (100)
%   - Number of simulated frames to use when computing motion covariance
% noiseModel                (constant variance of 8e-3 on a 0-1 scale)
%   - Noise model to use (see class in NoiseModel.m)
% saveCrop                  ({1:vr.Height, 1:vr.Width})
%   - Save only a crop of the motion, whole input image is used regardless
%     Format is a two-element cell array with first element specifying 
%     vertical crop and second specifying horizontal crop.
% scaleVideo                (1, no scaling)
%   - Scale frames of video before any processing
% sigma                     (3 px)            
%   - Amount of spatial smoothing (in px)
% temporalFilter            (No filtering) 
%   - What temporal filter to use, specifiying a temporal filter 
%     will load all phases into memory resulting in slower performance.
%     This function should process the third dimension of an input array
%     and take only one argument. 
% useFrames                 ([1, vr.NumberOfFrames])
%   - A tuple specifying a subset of frames to process
% usePyramidScales          (1:2)
%   - Which pyramid scales to use in the analysis. Level 1 is the
%     highest frequency level
% useMex                    (true)
%   - Whether to use mex files for faster motion computation
%
% Author(s): Neal Wadhwa
% License: Please refer to LICENSE file
% Date: August 2016
function [motion] = computeMotionAndCovarianceudtcwt( vr, varargin )
    p = inputParser();

    defaultCovarianceSamples = 100;
    defaultNoiseModel = NoiseModel.GetConstantNoiseModel(8e-3);
    defaultComputeCovarianceForEveryFrame = false;
    frameSize = size(vr.read(1));
    defaultCrop = {1:frameSize(1), 1:frameSize(2)};
    defaultScalesToUse = 1:2;
    defaultSigma = 0;
    defaultScale = 1;
    defaultFrames = [1, vr.NumberOfFrames];
    defaultTemporalFilter = [];
    defaultUseMex = true;
    
    addOptional(p, 'covarianceSamples', defaultCovarianceSamples, @isnumeric)
    addOptional(p, 'usePyramidScales', defaultScalesToUse)
    addOptional(p, 'noiseModel', defaultNoiseModel, @(x) isa(x, 'NoiseModel'))
    addOptional(p, 'computeCovarianceForEveryFrame', defaultComputeCovarianceForEveryFrame)
    addOptional(p, 'sigma', defaultSigma, @isnumeric);   
    addOptional(p, 'temporalFilter', defaultTemporalFilter);
    addOptional(p, 'scaleVideo', defaultScale);
    addOptional(p, 'useFrames', defaultFrames);
    addOptional(p, 'saveCrop', defaultCrop);
    addOptional(p, 'useMex', defaultUseMex);
    
    parse(p, varargin{:});
    
    covarianceSamples  = p.Results.covarianceSamples;
    usePyramidScales   = p.Results.usePyramidScales;
    noise_model        = p.Results.noiseModel;
    sigma              = p.Results.sigma;
    temporalFilter     = p.Results.temporalFilter;
    scale_factor       = p.Results.scaleVideo;
    frameRange         = p.Results.useFrames;
    crop               = p.Results.saveCrop;
    computeCovarianceForEveryFrame = p.Results.computeCovarianceForEveryFrame;
    useMex             = p.Results.useMex;
%     temporalFilter=[];
    if (nargout == 2)
        computeCovariance = true;
    else
        computeCovariance = false;
    end

   [Faf, Fsf] = NDAntonB2; %(Must use ND filters for both)
   [af, sf] = NDdualfilt1;

    computeMotionFunction = @(a, b, c, d, e) computeMotionFromPhaseudtcwt(a, b, c, d, e, useMex);

    
    % Compute displacement relative to first frame, temporally filtered
    % If the filter has no DC, then the displacement will really be
    % relative to the mean  location
    %readFrame = @(k) imresize(rgb2y(im2single(vr.read(frameRange(1)+k-1))), scale_factor);
    
    
    nF = frameRange(2)-frameRange(1)+1;    
    
    for frameIDX = 1:nF
        ffframe = rgb2ntsc(im2single(vr.read(frameIDX+frameRange(1)-1)));
        fframe(:,:,frameIDX) = ffframe(:,:,1);
    end
    [h_vid, w_vid, ~] = size(fframe);
    h = numel(crop{1});    
    w = numel(crop{2});
    if (h > h_vid)
        h = h_vid;
        crop{1} = 1:h;
    end
    if (w > w_vid)
        w = w_vid;
        crop{2} = 1:w;
    end
   
    % Pyramid is hardcoded to have four orientations
    orientations = 6; 
    level = 4;
    uselevel = 2;
    % Determine which levels of the pyramid need to be computed
    % Saves time by not computing some levels
    levelsToCompute = [];
    for s = usePyramidScales
       levelsToCompute = [levelsToCompute 1+orientations*(s-1)+(1:orientations)];

    end
    
    %buildPyr = octave4PyrFunctionsFullSize(h, w, levelsToCompute);
    %提取ht最大层数，仅用usePyramid那几层（即前两层）
    % We cannot use conventional discerete derivative operators, such as 
    % [-1, 0, 1], because the filter responses of pyramid levels are 
    % high-frequency, while those operators are accurate for image with
    % most of their energy at low frequencies. Instead, we demodulate the
    % filter responses by dividing by a sinusoid of frequency equal to 
    % the filter's maximumal frequency response. The result is a
    % low-frequency image multiplied by a sinusoid, the first of which we
    % can take the derivative numerically and the second analytically.
    % See Fleet, "Measurement of Image Velocity." 1992

    % Compute the maximal frequency response and modulation for each
    % filter response.
        
    %referencePyramid = NDxWav2DMEX(fframe(:,:,1), level, Faf, af, 1);%buildPyr(readFrame(1));   
    
    tuningFrequencies = peakTuningFrequencyFullSizeudtcwt([h, w]); %level更改需要到函数内部改
    
    modulation = zeros(h, w, orientations*uselevel, 'single');    %size(referencePyramid, 3)
    for bandIDX = 1:orientations*uselevel     %size(referencePyramid, 3)
        tempTuningFrequency = adjustTuningFrequency(tuningFrequencies(bandIDX,:), [h ,w]);
        modulation(:,:,bandIDX) = computeModulation(tempTuningFrequency, [h, w]);        
    end
    
    % We unwrap and apply temporal filters online 
    if (isempty(temporalFilter))
        motion = computeMotionOnline( fframe, h, w, nF, crop, sigma, modulation, tuningFrequencies, computeMotionFunction, level, uselevel, orientations );
        temporalFilterCovarianceReduction = 1;
    else
        motion = computeMotionAllAtOnce( fframe, temporalFilter, h, w, nF, crop, sigma, modulation, tuningFrequencies, computeMotionFunction, level, uselevel, orientations);
        temporalFilterCovarianceReduction = ...
            computeTemporalFilterReduction(temporalFilter, nF);
    end         
    

%     if (computeCovariance)
%         if (computeCovarianceForEveryFrame)
%             motion_covariance = zeros(h, w, 2, 2, nF,'single');
%             for frameIDX = 1:nF
%                 motion_covariance(:,:,:,:,frameIDX) = computeCovarianceViaSimulationudtcwt(...
%                 fframe(:,:,frameIDX), covarianceSamples, noise_model, sigma,...
%                 modulation, tuningFrequencies, ...
%                 computeMotionFunction, level, uselevel, orientations);
%             end
%         else
%             motion_covariance = computeCovarianceViaSimulationudtcwt(...
%             fframe(:,:,2), covarianceSamples, noise_model, sigma,...
%             modulation, tuningFrequencies, ...
%             computeMotionFunction, level, uselevel, orientations);
%         end

%        motion_covariance = temporalFilterCovarianceReduction* ...
%             motion_covariance(crop{1}, crop{2},:,:,:);        
%    end
end


function motion = computeMotionOnline( readFrame, h, w, nF, crop, sigma, modulation, tuningFrequencies, computeMotionFunction, level, uselevels, orientations)
    % Initialize motion
    motion = zeros(h,w,2, nF,'single');    
   [Faf, Fsf] = NDAntonB2; %(Must use ND filters for both)
   [af, sf] = NDdualfilt1;
    referencePyramid = NDxWav2DMEX(readFrame(:,:,1), level+1, Faf, af, 1); %buildPyr(readFrame(1));  

        for uselevel = 1:uselevels
            for nothinggood = 1:2
                for originala = 1:3
                    pyr1 = i*referencePyramid{uselevel}{2}{nothinggood}{originala}+referencePyramid{uselevel}{1}{nothinggood}{originala};
                    %pyramp{uselevel}{1}{nothinggood}{originala} = abs(pyr1);
                    %pyramppha{uselevel}{2}{nothinggood}{originala} = angle(pyr1);
                    k = originala + (nothinggood-1)*3 + (uselevel-1)*6
                    previousPhase(:,:,k) = angle(pyr1);  %pyramppha{uselevel}{2}{nothinggood}{originala};
                    %previousPhase(:,:,k) = Unwrap_TIE_DCT_Iter(previousPhase(:,:,k));
                end
            end
        end

    %previousPhase = Unwrap_TIE_DCT_Iter(previousPhase);
    accumulatedPhase = zeros(h, w,...
       uselevels*orientations, 'single');

    for frameIDX = 2:nF
        
        currentFrame = readFrame(:,:,frameIDX);
        currentPyramid = NDxWav2DMEX(currentFrame, level+1, Faf, af, 1); %buildPyr(currentFrame);
        for uselevel = 1:uselevels
            for nothinggood = 1:2
                for originala = 1:3
                    pyr1 = i*currentPyramid{uselevel}{2}{nothinggood}{originala}+currentPyramid{uselevel}{1}{nothinggood}{originala};
                    %pyramppha{uselevel}{1}{nothinggood}{originala} = abs(pyr1);
                    %pyramppha{uselevel}{2}{nothinggood}{originala} = angle(pyr1);
                    k = originala + (nothinggood-1)*3 + (uselevel-1)*6;
                    currentPhase(:,:,k) =  angle(pyr1); %pyramppha{uselevel}{2}{nothinggood}{originala};
                    %currentPhase(:,:,k) = Unwrap_TIE_DCT_Iter(currentPhase(:,:,k));
                    currentPyramidd(:,:,k) = pyr1;
                end
            end
        end
   
        
        fprintf('Computing motion for frame %d\n', frameIDX);              
           %currentPhase = Unwrap_TIE_DCT_Iter(currentPhase);
        accumulatedPhase = accumulatedPhase + mod(pi+currentPhase - previousPhase, 2*pi)-pi;                
        previousPhase = currentPhase;                
                        
        phaseDifference = accumulatedPhase;
        
        % Perform OLS on constraints to get flow estimate at point
        current_motion = computeMotionFunction(phaseDifference, ...
            currentPyramidd, sigma, modulation, tuningFrequencies);
        motion(:,:,:,frameIDX) = current_motion(crop{1}, crop{2},:);       
    end 

end

function motion = computeMotionAllAtOnce( readFrame, temporalFilter, h, w, nF, crop, sigma, modulation, tuningFrequencies, computeMotionFunction, level, uselevels, orientations )
    % Initialize motion
    motion = zeros(h,w,2, nF,'single');    
    
   [Faf, Fsf] = NDAntonB2; %(Must use ND filters for both)
   [af, sf] = NDdualfilt1;
    referencePyramid = NDxWav2DMEX(readFrame(:,:,1), level+1, Faf, af, 1); %buildPyr(readFrame(1));  

        for uselevel = 1:uselevels
            for nothinggood = 1:2
                for originala = 1:3
                    pyr1 = i*referencePyramid{uselevel}{2}{nothinggood}{originala}+referencePyramid{uselevel}{1}{nothinggood}{originala};
                    %pyramp{uselevel}{1}{nothinggood}{originala} = abs(pyr1);
                    %pyramppha{uselevel}{2}{nothinggood}{originala} = angle(pyr1);
                    k = originala + (nothinggood-1)*3 + (uselevel-1)*6
                    previousPhase(:,:,k) = angle(pyr1); %pyramppha{uselevel}{2}{nothinggood}{originala};
                    %previousPhase(:,:,k) = un2wrap(previousPhase(:,:,k)); %Unwrap_TIE_DCT_Iter(previousPhase(:,:,k));
                end
            end
        end
 
    accumulatedPhase = zeros(size(previousPhase,1), size(previousPhase,2),...
       uselevels*orientations, nF, 'single');
    %accumulatedPhase = zeros(size(referencePyramid,1), size(referencePyramid,2),...
     %   size(referencePyramid,3), nF, 'single');
    
    for frameIDX = 2:nF
        fprintf('Computing phase for frame %d\n', frameIDX);              

        currentFrame = readFrame(:,:,frameIDX);
        currentPyramid = NDxWav2DMEX(currentFrame, level+1, Faf, af, 1); %buildPyr(currentFrame);
        for uselevel = 1:uselevels
            for nothinggood = 1:2
                for originala = 1:3
                    pyr1 = i*currentPyramid{uselevel}{2}{nothinggood}{originala}+currentPyramid{uselevel}{1}{nothinggood}{originala};
                    %pyramppha{uselevel}{1}{nothinggood}{originala} = abs(pyr1);
                    %pyramppha{uselevel}{2}{nothinggood}{originala} = angle(pyr1);
                    k = originala + (nothinggood-1)*3 + (uselevel-1)*6;
                    currentPhase(:,:,k) = angle(pyr1);
                    
                    %currentPhase(:,:,k) = Unwrap_TIE_DCT_Iter(currentPhase(:,:,k));
                    currentPyramidd(:,:,k) = pyr1;
                end
            end
        end
%          3
%         currentPhase = angle(currentPyramid);
%         currentPhase = un2wrap(currentPhase); %Unwrap_TIE_DCT_Iter(currentPhase);

        accumulatedPhase(:,:,:,frameIDX) = accumulatedPhase(:,:,:,frameIDX-1) + mod(pi+currentPhase - previousPhase, 2*pi)-pi;                
        previousPhase = currentPhase;                
    end

    % S = whos('accumulatedPhase');
    % fprintf('Accumulated Phase 峰值内存占用: %.2f GB\n', S.bytes / (1024^3));



    
    for levelIDX = 1:size(accumulatedPhase, 3)
        
        fprintf('Temporally filtering level %d\n', levelIDX);
        % Necessary to use squeeze to move time to dimension 3
        accumulatedPhase(:,:,levelIDX,:) = ...
            temporalFilter(squeeze(accumulatedPhase(:,:,levelIDX,:)));
    end
    
    for frameIDX = 1:nF
        
        fprintf('Computing motion for frame %d\n', frameIDX);   
        phaseDifference = accumulatedPhase(:,:,:,frameIDX);
        
        % Perform OLS on constraints to get flow estimate at point
        current_motion = computeMotionFunction(phaseDifference, ...
            currentPyramidd, sigma, modulation, tuningFrequencies);
        % 相位残差多方向、当前滤波器（该层该方向）与图片相乘值（响应）、
        % 峰值频率是滤波器对二位脉冲响应的频率值（直接用就行，导入无图像的滤波器）

%各方向上处理

        motion(:,:,:,frameIDX) = current_motion(crop{1}, crop{2},:);
        
    end 

end
function phase = un2wrap(phase)
 a=phase;                                             %将包裹相位ph赋值给a
    %下面开始进行最小二乘解包裹运算
    [M,N]=size(a);                                    %计算二维包裹相位的大小（行、列数）
    dx=zeros(M,N);dy=zeros(M,N);                      %预设包裹相位沿x方向和y方向的梯度
    m=1:M-1; 
    dx(m,:)=a(m+1,:)-a(m,:);                          %计算包裹相位沿x方向的梯度
    dx=dx-pi*round(dx/pi);                            %去除梯度中的跳跃
    n=1:N-1;
    dy(:,n)=a(:,n+1)-a(:,n);                          %计算包裹相位沿y方向的梯度
    dy=dy-pi*round(dy/pi);                            %去除梯度中的跳跃
    p=zeros(M,N);p1=zeros(M,N);p2=zeros(M,N); %为计算ρnm作准备
    m=2:M;
    p1(m,:)=dx(m,:)-dx(m-1,:);                        %计算Δgxnm-Δgx(n-1)m
    n=2:N;
    p2(:,n)=dy(:,n)-dy(:,n-1);                        %计算Δgynm–Δgyn(m-1)
    p=p1+p2;                                          %计算ρnm
    p(1,1)=dx(1,1)+dy(1,1);                           %计算ρnm
    n=2:N;
    p(1,n)=dx(1,n)+dy(1,n)-dy(1,n-1);                 %赋值Neumann边界条件
    m=2:M;
    p(m,1)=dx(m,1)-dx(m-1,1)+dy(m,1);
    pp=dct2(p)+eps;                                   %计算ρnm的DCT
    fi=zeros(M,N);
    for m=1:M                                         %计算Φnm在DCT域的精确解
       for n=1:N  
          fi(m,n)=pp(m,n)/(2*cos(pi*(m-1)/M)+2*cos(pi*(n-1)/N)-4+eps);
       end
    end
    fi(1,1)=pp(1,1);                                  %赋值DCT域的Φ11
    phase=idct2(fi);
end