% EDGES = GETEDGES(IM, THRESHOLD, LEVELSTOUSE)
% 
% Determines pixels in IM that correspond to edges. Returns a binary mask
% EDGES that shows these locations. Works by constructing a laplacian
% pyramid of the input image and then computing the local energy at each
% point. This energy is summed across levels specified in LEVELSTOUSE and
% then places where the energy is greater than threshold are marked as
% edges.
%
% This function will treat textured areas as edge areas, which is the
% desired behavious since the output is used to mask out pixels that can't
% be used in noise analysis.
% 
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
%
function [ edges ] = getEdges( im, threshold, levelsToUse )
    if (nargin < 2)
       threshold = 0.1; 
    end
    
    if (nargin < 3)
       levelsToUse = 2:3; 
    end
    
    im = im2double(rgb2gray(im));
        [pyr, pind] = buildLpyr(im);
    
    kernel = [1 2 1; 2 4 2; 1 2 1]/4;

    totalEdgeEnergy = zeros(size(im));
    for levelIDX = 1:numel(levelsToUse);
       currentLevel = pyrBand(pyr,pind, levelsToUse(levelIDX));
       powerMap = sqrt(imfilter(currentLevel.^2, kernel));
       totalEdgeEnergy = totalEdgeEnergy + imresize(powerMap, size(im));
    end
    totalEdgeEnergy = totalEdgeEnergy./numel(levelsToUse);
    
    edges = totalEdgeEnergy>threshold;    
end

