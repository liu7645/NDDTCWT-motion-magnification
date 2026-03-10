% MAXVAL, YLOC, XLOC = MAX2D(IM)
% 
% Computes the maximum value and location in a 2D image.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function [ maxVal,yLoc,xLoc ] = max2D( im )
    [maxVal,yLoc] = max(im);
    [maxVal,xLoc] = max(maxVal);
    yLoc = yLoc(xLoc);
end

