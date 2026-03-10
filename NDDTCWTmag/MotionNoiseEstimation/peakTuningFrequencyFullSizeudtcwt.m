% TUNINGFREQUENCY = PEAKTUNINGFREQUENCYFULLSIZE(BUILDPYRAMID, IMSIZE) 
% 
% TUNINGFREQUENCY is the frequency, for which each filter in BUILDPYRAMID 
% maximally selects for. The frequency is only found to integer precision
% and is a two dimensional coordinate. The center of the image is
% considered (0,0) with positive coordinates in the bottom and right and
% negative coordinates in the top and left.
%
% Expects buildPyramid to come from the function octave4PyrFunctionsFullSize

function tuningFrequency = peakTuningFrequencyFullSizeudtcwt(imSize)
    % Create im, an image with a delta function at its center
    im = zeros(imSize);
    ctr = ceil((imSize+1)/2);
    im(ctr(1),ctr(2)) = 1;
    ht = 4;
    numeloffilter = ht * 2;
   [Faf, Fsf] = NDAntonB2; %(Must use ND filters for both)
   [af, sf] = NDdualfilt1;
    % Find the impulse response of each level of the pyramid
    pyr = NDxWav2DMEX(im, ht+1, Faf, af, 1); %buildPyramid(im);
    % There is a tuning frequency for each pyramid level
    tuningFrequency = zeros(numeloffilter, 2);

    %编写nddtcwt相位
    for level = 1:2
            for nothinggood = 1:2
    %for k = 1:size(tuningFrequency,1)
                for k = 1:3
                    
        impulseResponse = i*pyr{level}{2}{nothinggood}{k}+pyr{level}{1}{nothinggood}{k};
       %impulseResponse = pyr(:,:,k);

       freqResponse = abs(fft2(impulseResponse));        
       [~, yLoc, xLoc] = max2D(freqResponse);     
       
       % In frequency plane, frequencies greater than the nyquist rate, 
       % should be negative, 
       [Ysz, Xsz] = size(impulseResponse);
       nyquistY = floor(Ysz/2);
       nyquistX = floor(Xsz/2);
       k2 = k + (nothinggood-1)*3 + (level-1)*6;
       tuningFrequency(k2,1) = mod(nyquistX + xLoc, Xsz) - nyquistX;
       tuningFrequency(k2,2) = mod(nyquistY + yLoc, Ysz) - nyquistY;
       
                end
            end
    end
end