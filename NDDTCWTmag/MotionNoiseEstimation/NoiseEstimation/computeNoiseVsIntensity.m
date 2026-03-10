% NOISEMODEL = COMPUTENOISEVSINTENSITY(GRAYFRAME, VARIANCEFRAME, BINS)
%
% Estimates a signal-dependent NOISEMODEL given corresponding mean and
% variance values from an image. The intensity range is divided into BINS
% bins and the mean variance in each intensity range bin is marked as the
% variance of that bin. More BINS means higher resolution, but more noise
% in the noise model.
function [ noiseModel] = computeNoiseVsIntensity( grayFrame, varianceFrame, bins )
    binEdges = linspace(0,1,bins+1);
    binCenters= binEdges(1:end-1)+1/(2*bins);
    
    varianceValues = zeros(bins,1);
    for k = 1:bins
       lowerBinBound = binEdges(k);
       upperBinBound = binEdges(k+1); 
       idx = and(lowerBinBound < grayFrame, grayFrame < upperBinBound);
       VV = varianceFrame(idx);
       varianceValues(k) = mean(VV);        
    end
    noiseModel = NoiseModel(true, varianceValues, binCenters);

end

