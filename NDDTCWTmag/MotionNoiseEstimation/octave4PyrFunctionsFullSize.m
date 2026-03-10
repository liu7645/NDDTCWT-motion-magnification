% BUILDPYR = OCTAVE4PYRFUNCTIONSFULLSIZE(HEIGHT, WIDTH, LEVELS)
%
% Creates a function that creates a four orientation octave bandwidth
% complex steerable pyramid that computes LEVELS of the pyramid. 
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function [ buildPyr] = octave4PyrFunctionsFullSize(height, width, levels)    
    ht = maxSCFpyrHt(zeros(height,width));
    filters = getFilters([height width], 2.^[0:-1:-ht], 4);
    if (nargin == 2)
        levels = 1:numel(filters);
    end
    [~, filtIDX] = getFilterIDX(filters);
        for k = 1:numel(filters)        
        if (any(abs(levels-k)<1e-10))
            filters{k} = filters{k} *height*width/(numel(filtIDX{k,1})*numel(filtIDX{k,2}));
        end
    end 
    filters = filters(levels);
    buildPyr = @(im) buildSCFpyrGenSameSize(im, filters) ;
end

