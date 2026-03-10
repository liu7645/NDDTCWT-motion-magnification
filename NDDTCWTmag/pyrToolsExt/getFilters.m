% FILTERS = getFilters(DIMENSION, RVALS, ORIENTATIONS, ...)
%
% Returns the transfer function of the different scales and orientations of
% a complex steerable pyramid.
%
% DIMENSIONS is the dimension of the image to be filtered
% RVALS specify the boundary between adjacent filters
% ORIENTATIONS specify the number of orientations
%
% Optional Arguments
% TWIDTH controls the falloff of the filters
%
% Based on buildSCFpyr in matlabPyrTools
%
% Authors: Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: July 2013
%

function filters = getFilters(dimension, rVals, orientations, varargin )

p = inputParser;

defaultTwidth = 1; %Controls falloff of filters

addRequired(p, 'dimension');
addRequired(p, 'rVals');
addRequired(p, 'order');
addOptional(p, 'twidth', defaultTwidth, @isnumeric);
parse(p, dimension, rVals, orientations, varargin{:});

dimension = p.Results.dimension;
rVals = p.Results.rVals;
orientations = p.Results.order;
twidth = p.Results.twidth;


[angle, log_rad] = getPolarGrid(dimension); % Get polar coordinates of frequency plane
%save('angle.mat','angle')
%save('log_rad.mat','log_rad')
count = 1;
[himask, lomaskPrev] = getRadialMaskPair(rVals(1), log_rad, twidth);%ÂË²¨Æ÷±ß½ç¡¢½Ç¶È¡¢Í»È»Ë¥¼õ
%save('himask.mat','himask')
%save('lomaskPrev.mat','lomaskPrev')
filters{count} = himask;
count = count + 1;
for k = 2:max(size(rVals))
   [himask, lomask] = getRadialMaskPair(rVals(k), log_rad, twidth);
   radMask = himask.*lomaskPrev;
   %save('radMask.mat','radMask')
   for j = 1:orientations
      anglemask = getAngleMask(j, orientations, angle);
      %save('anglemask.mat','anglemask')
      filters{count} = radMask.*anglemask/2;
      count = count + 1;
   end
   
   lomaskPrev = lomask;
end
filters{count} = lomask;

end

