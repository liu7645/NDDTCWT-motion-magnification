% NOISY_IMAGE = SIMULATENOISE(IMAGE, NOISEMODEL)
%
% Creates a noisy version of IMAGE based on the parameters in noiseModel.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function noisy_image = SimulateNoise(image, noiseModel)
    if (noiseModel.signalDependentNoiseModel)        
        noise_stddev_image = sqrt(max(interp1(noiseModel.binCenters, noiseModel.variance, image, 'linear', 'extrap'),0));
        noisy_image = image + randn(size(image)).*noise_stddev_image;
    else
        noisy_image = image + randn(size(image))* sqrt(noiseModel.variance);
    end
end

