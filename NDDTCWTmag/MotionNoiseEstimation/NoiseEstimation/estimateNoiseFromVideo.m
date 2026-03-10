% [NOISEMODEL, GRAYFRAME, VARIANCEFRAME] = ESTIMATENOISEFROMVIDEO(VR, FRAMERANGE, EDGETHRESHOLD, BINS )
%
% Estimates a signal-dependent noise model from a video. Expects the input
% video to be a class with properties Width, Height, NumberOfFrames and a
% method read that returns frames of the video. See VideoReader.m for an
% example. The input video should be mostly static. Edges are detected and
% the noise level is estimated from the remaining flat regions. See
% getEdges for the meaning of edgeThreshold and computeNoiseVsIntensity for
% the meaning of bins.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
%
function [noiseModel, grayFrame, varianceFrame] = estimateNoiseFromVideo( vr, frameRange, edgeThreshold, bins, levelsToUse )
    if (nargin < 3);
        edgeThreshold = 0.05;
    end
    if (nargin < 4)
       bins = 128; 
    end
    if (nargin < 5)
        levelsToUse = 2:3;
    end
    
    meanFrame = onlineMean(vr, frameRange);
    varianceFrame = onlineVariance(vr, meanFrame, frameRange,true);
    grayFrame = rgb2y(meanFrame);
    
    edges = getEdges(vr.read(1), edgeThreshold, levelsToUse);
    notEdges = ~edges;
    
    noiseModel = computeNoiseVsIntensity(grayFrame(notEdges), varianceFrame(notEdges), bins);            
end

