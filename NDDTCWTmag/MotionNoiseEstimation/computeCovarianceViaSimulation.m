% Computes noise variance of motion empirically, Takes a frame and adds
% noise according to the parameters and then computes the covariance of the
% result over samples.
% 
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function [ motion_covariance ] = computeCovarianceViaSimulation(im1, covarianceSamples, noiseModel, sigma, modulation, tuningFrequencies, buildPyr, computeMotion)
    fprintf('Computing motion variance through simulation.\n');
    motion = zeros(size(im1,1), size(im1,2), 2, covarianceSamples,'single');
    for k = 1:covarianceSamples
        fprintf('Covariance Simulation: %d/%d\n', k, covarianceSamples);
        A1 = SimulateNoise(im1, noiseModel);
        A2 = SimulateNoise(im1, noiseModel);
        
        pyramid1 = buildPyr(A1);
        pyramid2 = buildPyr(A2);
        phaseDifference = mod(pi+angle(pyramid2)-angle(pyramid1),2*pi)-pi;                
        motion(:,:,:,k) = computeMotion(phaseDifference, ...
            pyramid1, sigma, modulation, tuningFrequencies);   
    end
    
    % Compute variance and covariances of horizontal and vertical motion
    % components
    meanFlow = mean(motion,4);
    motion = bsxfun(@minus, motion, meanFlow);
    [h, w, ~] = size(im1);
    motion_covariance = zeros(h, w, 2, 2, 'single');
    for k = 1:2
        for j = k:2       
            motion_covariance(:,:,k,j) = mean(motion(:,:,k,:).*motion(:,:,j,:),4);
            if (k~=j)
                motion_covariance(:,:,j,k) = motion_covariance(:,:,k,j);
            end               
        end
    end

end

