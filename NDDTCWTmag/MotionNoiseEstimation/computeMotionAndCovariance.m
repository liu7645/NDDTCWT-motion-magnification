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
function [motion, motion_covariance] = computeMotionAndCovariance( vr, varargin )
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
%         temporalFilter=[];
    computeCovariance = false;
    if (nargout == 2)
        computeCovariance = true;
    end


    computeMotionFunction = @(a, b, c, d, e) computeMotionFromPhase(a, b, c, d, e, useMex);
        
    
    % Compute displacement relative to first frame, temporally filtered
    % If the filter has no DC, then the displacement will really be
    % relative to the mean  location
    readFrame = @(k) imresize(rgb2y(im2single(vr.read(frameRange(1)+k-1))), scale_factor);
    
    
    nF = frameRange(2)-frameRange(1)+1;    
    [h_vid, w_vid, ~] = size(readFrame(1));
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
    orientations = 4; 
    
    % Determine which levels of the pyramid need to be computed
    % Saves time by not computing some levels
    levelsToCompute = [];
    for s = usePyramidScales
       levelsToCompute = [levelsToCompute 1+orientations*(s-1)+(1:orientations)];
    end
    buildPyr = octave4PyrFunctionsFullSize(h, w, levelsToCompute);
    
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
    referencePyramid = buildPyr(readFrame(1));   
    tuningFrequencies = peakTuningFrequencyFullSize(buildPyr, [h, w]);
    modulation = zeros(h, w, size(referencePyramid, 3), 'single');    
    for bandIDX = 1:size(referencePyramid, 3)
        tempTuningFrequency = adjustTuningFrequency(tuningFrequencies(bandIDX,:), [h ,w]);
        modulation(:,:,bandIDX) = computeModulation(tempTuningFrequency, [h, w]);        
    end
    
    % We unwrap and apply temporal filters online 
    if (isempty(temporalFilter))
        motion = computeMotionOnline( readFrame, buildPyr, h, w, nF, crop, sigma, modulation, tuningFrequencies, computeMotionFunction );
        temporalFilterCovarianceReduction = 1;
    else
        motion = computeMotionAllAtOnce( readFrame, buildPyr, temporalFilter, h, w, nF, crop, sigma, modulation, tuningFrequencies, computeMotionFunction );
        temporalFilterCovarianceReduction = ...
            computeTemporalFilterReduction(temporalFilter, nF);
    end         
    
    
    
    if (computeCovariance)
        if (computeCovarianceForEveryFrame)
            motion_covariance = zeros(h, w, 2, 2, nF,'single');
            for frameIDX = 1:nF
                motion_covariance(:,:,:,:,frameIDX) = computeCovarianceViaSimulation(...
                readFrame(frameIDX), covarianceSamples, noise_model, sigma,...
                modulation, tuningFrequencies, buildPyr, ...
                computeMotionFunction);
            end
        else
            motion_covariance = computeCovarianceViaSimulation(...
            readFrame(2), covarianceSamples, noise_model, sigma,...
            modulation, tuningFrequencies, buildPyr, ...
            computeMotionFunction);
        end

       motion_covariance = temporalFilterCovarianceReduction* ...
            motion_covariance(crop{1}, crop{2},:,:,:);        
    end
end


function motion = computeMotionOnline( readFrame, buildPyr, h, w, nF, crop, sigma, modulation, tuningFrequencies, computeMotionFunction )
    % Initialize motion
    motion = zeros(h,w,2, nF,'single');    
    
    referencePyramid = buildPyr(readFrame(1));   
    previousPhase = angle(referencePyramid);
    accumulatedPhase = zeros(h,w,size(referencePyramid,3),'single');
    for frameIDX = 2:nF
        
        currentFrame = readFrame(frameIDX);
        currentPyramid = buildPyr(currentFrame);
        
        fprintf('Computing motion for frame %d\n', frameIDX);              
        currentPhase = angle(currentPyramid);
        accumulatedPhase = accumulatedPhase + mod(pi+currentPhase - previousPhase, 2*pi)-pi;                
        previousPhase = currentPhase;                
                        
        phaseDifference = accumulatedPhase;
        
        % Perform OLS on constraints to get flow estimate at point
        current_motion = computeMotionFunction(phaseDifference, ...
            currentPyramid, sigma, modulation, tuningFrequencies);
        motion(:,:,:,frameIDX) = current_motion(crop{1}, crop{2},:);       
    end 

end

function motion = computeMotionAllAtOnce( readFrame, buildPyr, temporalFilter, h, w, nF, crop, sigma, modulation, tuningFrequencies, computeMotionFunction )
    % Initialize motion
    motion = zeros(h,w,2, nF,'single');    
    
    referencePyramid = buildPyr(readFrame(1));   
    previousPhase = angle(referencePyramid);
    accumulatedPhase = zeros(size(referencePyramid,1), size(referencePyramid,2),...
        size(referencePyramid,3), nF, 'single');
    
    for frameIDX = 2:nF
        fprintf('Computing phase for frame %d\n', frameIDX);              

        currentFrame = readFrame(frameIDX);
        currentPyramid = buildPyr(currentFrame);
        
        currentPhase = angle(currentPyramid);
        accumulatedPhase(:,:,:,frameIDX) = accumulatedPhase(:,:,:,frameIDX-1) +...
            mod(pi+currentPhase - previousPhase, 2*pi)-pi;                
        previousPhase = currentPhase;                
    end
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
            currentPyramid, sigma, modulation, tuningFrequencies);
        motion(:,:,:,frameIDX) = current_motion(crop{1}, crop{2},:);
               
    end 

end