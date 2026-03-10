% PYR = buildSCFpyrGenPar(IM, FILTERS, ...)
%
% This is pyramid building function, which will apply FILTERS to the image
% and give back a pyramid. Unlike buildSCFpyrGen, this function returns a
% pyramid of levels that are all the same size.
%
% Based on buildSCFpyrGen
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
%
function [pyr] = buildSCFpyrGenSameSize(im, filters, varargin)

    nFilts = max(size(filters));

    % Return pyramid in the usual format of a stack of column vectors
    imdft = fftshift(fft2(im)); %DFT of image

    pyr = zeros(size(im, 1), size(im, 2), numel(filters), 'single');
    for k = 1:nFilts      
        tempDFT = filters{k}.*imdft; % Transform domain                           
        curResult = ifft2(ifftshift(tempDFT));    
        pyr(:,:,k) = curResult;
    end
end
