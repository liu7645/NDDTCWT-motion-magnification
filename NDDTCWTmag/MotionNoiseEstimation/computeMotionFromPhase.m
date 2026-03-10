%  MOTION = COMPUTEMOTIONHALIDE(PHASEDIFFERENCES, FILTERRESPONSE, SIGMA, MODULATION, TUNINGFREQ)
% 
% Wrapper around Halide code to compute the flow. The computation of 
% filter respones and phase differences are computed outside of Halide
% since these things are already pretty fast thanks to fftw. 
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function motion = computeMotionFromPhase(phaseDifferences, filterResponse, sigma, modulation, tuningFreq, useMex)
    if (nargin < 6)
        useMex = true;
    end
    motion = zeros(size(phaseDifferences, 1), size(phaseDifferences, 2), 2, 'single');
    if (and(useMex, exist('computeMotionHalideMex', 'file') == 3)) % Does the mex file exist?
        computeMotionHalideMex(phaseDifferences, real(filterResponse), ...
            imag(filterResponse), single(real(modulation)), ...
            single(imag(modulation)), single(tuningFreq), ...
            sigma, motion);
    else
        % Difficult to setup Halide on windows, so Matlab code instead
        % Precompute derivative filter kernels. Uses five tap kernels
        % specified in Simoncelli, "DESIGN OF MULTI-DIMENSIONAL DERIVATIVE 
        % FILTERS" 1994, see computeMotionHalideMex.cpp for more details
        width = size(phaseDifferences, 2);
        height = size(phaseDifferences, 1);
        derivKernel = [0.109604, 0.276691, 0.0, -0.276691, -0.109604];
        derivPrefilter = [0.037659, 0.249153, 0.426375, 0.249153, 0.037659];
        spaceBlurExtent = 2*ceil(sigma);
        x = -spaceBlurExtent:spaceBlurExtent;
        blurKernel = exp(-(x.*x)/(2*sigma.^2+eps));
        blurKernel = blurKernel./sum(blurKernel);
        
        weightSq = abs(filterResponse).^2 + eps;
        for c = 1:size(tuningFreq,1)
            adjustedTuningFreq(c,:) = adjustTuningFrequency(tuningFreq(c,:), [height, width]);
        end
        
        lowpass = filterResponse.*conj(modulation);
        
        gradx = Differentiate(lowpass, derivKernel, derivPrefilter);        
        grady = Differentiate(lowpass, derivPrefilter, derivKernel);
        
        for c = 1:size(adjustedTuningFreq,1)
            gradx(:,:,c) = gradx(:,:,c).*modulation(:,:,c) + 1i*filterResponse(:,:,c)*adjustedTuningFreq(c,1);
            grady(:,:,c) = grady(:,:,c).*modulation(:,:,c) + 1i*filterResponse(:,:,c)*adjustedTuningFreq(c,2);
        end
        phixW = imag(conj(filterResponse) .* gradx);
        phiyW = imag(conj(filterResponse) .* grady);
        phix = 1./weightSq .* phixW;
        phiy = 1./weightSq .* phiyW;
        X11 = BlurAndSum(phixW, phix, blurKernel);
        X12 = BlurAndSum(phixW, phiy, blurKernel);
        X22 = BlurAndSum(phiyW, phiy, blurKernel);
        
        Y1 = BlurAndSum(phixW, phaseDifferences, blurKernel);
        Y2 = BlurAndSum(phiyW, phaseDifferences, blurKernel);
        
        D = 1./(X11.*X22 - X12.^2);
        motion(:,:,1) = (-X22.*Y1 + X12.*Y2).*D;
        motion(:,:,2) = (X12.*Y1 - X11.*Y2).*D;
    end
end

% Assumes kernel length of 5
function input = Differentiate(input, kernel_x, kernel_y)
    input = padarray(input, [2, 2], 'replicate', 'both');
    kernel_x = kernel_x(:)';
    kernel_y = kernel_y(:);
    input = convn(input, kernel_x, 'valid');
    input = convn(input, kernel_y, 'valid');
end

% Expects kernel to be odd-sized
function out = BlurAndSum(in1, in2, kernel1D)
    extent = (numel(kernel1D)-1)/2;
    out = in1.*in2;
    out = padarray(out, [extent, extent], 'replicate', 'both');
    out = convn(out, kernel1D(:)', 'valid');
    out = convn(out, kernel1D(:), 'valid');
    out = sum(out, 3);

end