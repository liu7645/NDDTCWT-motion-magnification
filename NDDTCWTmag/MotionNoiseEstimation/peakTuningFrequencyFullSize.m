% TUNINGFREQUENCY = PEAKTUNINGFREQUENCYFULLSIZE(BUILDPYRAMID, IMSIZE) 
% 
% TUNINGFREQUENCY is the frequency, for which each filter in BUILDPYRAMID 
% maximally selects for. The frequency is only found to integer precision
% and is a two dimensional coordinate. The center of the image is
% considered (0,0) with positive coordinates in the bottom and right and
% negative coordinates in the top and left.
%
% Expects buildPyramid to come from the function octave4PyrFunctionsFullSize

function tuningFrequency = peakTuningFrequencyFullSize(buildPyramid, imSize)
    % Create im, an image with a delta function at its center
    im = zeros(imSize);
    ctr = ceil((imSize+1)/2);
    im(ctr(1),ctr(2)) = 1;
    
    % Find the impulse response of each level of the pyramid
    pyr = buildPyramid(im);
    % There is a tuning frequency for each pyramid level
    tuningFrequency = zeros(size(pyr,3), 2);
    for k = 1:size(tuningFrequency,1)
       impulseResponse = pyr(:,:,k);
       freqResponse = abs(fft2(impulseResponse));        
       [~, yLoc, xLoc] = max2D(freqResponse);     
       
       % In frequency plane, frequencies greater than the nyquist rate, 
       % should be negative, 
       [Ysz, Xsz] = size(impulseResponse);
       nyquistY = floor(Ysz/2);
       nyquistX = floor(Xsz/2);
       tuningFrequency(k,1) = mod(nyquistX + xLoc, Xsz) - nyquistX;
       tuningFrequency(k,2) = mod(nyquistY + yLoc, Ysz) - nyquistY;
       
    end
end