% Computes noise variance of motion empirically, Takes a frame and adds
% noise according to the parameters and then computes the covariance of the
% result over samples.
% 
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function [ motion_covariance ] = computeCovarianceViaSimulation(im1, covarianceSamples, noiseModel, sigma, modulation, tuningFrequencies, computeMotion, level, uselevels, orientations)
    fprintf('Computing motion variance through simulation.\n');



    [Faf, Fsf] = NDAntonB2; %(Must use ND filters for both)
   [af, sf] = NDdualfilt1;
    motion = zeros(size(im1,1), size(im1,2), 2, covarianceSamples,'single');
    for k = 1:covarianceSamples
        fprintf('Covariance Simulation: %d/%d\n', k, covarianceSamples);
        A1 = SimulateNoise(im1, noiseModel);
        A2 = SimulateNoise(im1, noiseModel);
        referencePyramid = NDxWav2DMEX(A1, level+1, Faf, af, 1); %buildPyr(readFrame(1));  

        for uselevel = 1:uselevels
            for nothinggood = 1:2
                for originala = 1:3
                    pyr1 = i*referencePyramid{uselevel}{2}{nothinggood}{originala}+referencePyramid{uselevel}{1}{nothinggood}{originala};
                    pyramid1(:,:,k) = pyr1;
                end
            end
        end
        referencePyramid = NDxWav2DMEX(A2, level+1, Faf, af, 1); %buildPyr(readFrame(1));  

        for uselevel = 1:uselevels
            for nothinggood = 1:2
                for originala = 1:3
                    pyr2 = i*referencePyramid{uselevel}{2}{nothinggood}{originala}+referencePyramid{uselevel}{1}{nothinggood}{originala};
                    pyramid2(:,:,k) = pyr2;
                end
            end
        end
        
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

