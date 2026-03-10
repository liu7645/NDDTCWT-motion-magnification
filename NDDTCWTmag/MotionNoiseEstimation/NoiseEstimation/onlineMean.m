% MEANFRAME = ONLINEMEAN(VR, FRAMERANGE)
%
% Computes the mean frame (MEANFRAME) over time of the video specified by 
% VR over the times specified in the two element array FRAMERANGE.
% Does computation without loading video.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function [ meanFrame ] = onlineMean( vr, frameRange )
    meanFrame = zeros(size(vr.read(1)));
    numberOfFrames = frameRange(2)-frameRange(1) + 1;
    for ff = frameRange(1):frameRange(2)
        meanFrame = meanFrame + im2double(vr.read(ff));
        fprintf('Computing Mean: %d/%d\n', ff, frameRange(2));
    end
    meanFrame = meanFrame./numberOfFrames;
end

