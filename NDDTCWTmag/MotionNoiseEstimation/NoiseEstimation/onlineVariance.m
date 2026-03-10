% VARIANCEFRAME = ONLINEVARIANCE(VR, MEANFRAME, FRAMERANGE, GRAYSCALE)
%
% Computes the variance (VARIANCEFRAME) over time of the video specified by 
% VR over the times specified in the two element array FRAMERANGE. The 
% MEANFRAME should have been computed by onlineMean. If GRAYSCALE is true, 
% convert the MEANFRAME and frames of VR to grayscale prior to processing.
% Does computation without loading video.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function [ varianceFrame ] = onlineVariance( vr, meanFrame, frameRange, grayscale)
    varianceFrame = zeros(size(vr.read(1)));
    if (grayscale)
       varianceFrame = rgb2y(varianceFrame); 
       meanFrame = rgb2y(meanFrame);
    end
    numberOfFrames = frameRange(2)-frameRange(1) + 1;
    for ff = frameRange(1):frameRange(2)
       currentFrame = im2double(vr.read(ff));
       if (grayscale)
            currentFrame = rgb2y(currentFrame); 
        end
        varianceFrame = varianceFrame + (currentFrame - meanFrame).^2;
        fprintf('ComputingVariance: %d/%d\n', ff, frameRange(2));
    end
    varianceFrame = varianceFrame./(numberOfFrames -1 );

end

