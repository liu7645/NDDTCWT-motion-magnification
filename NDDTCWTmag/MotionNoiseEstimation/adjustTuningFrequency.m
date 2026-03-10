% TUNINGFREQUENCY = ADJUSTTUNINGFREQUENCY(TUNINGFREQUENCY, IMSIZE) 
% 
% The input TUNINGFREQUENCY is a tuple that is converted to a normalized 
% frequency tuple, in which each element is in the range [-pi, pi]. The
% input should come from peakTuningFrequencyFullSize or peakTuningFrequency
% See those functions for more details.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
%
function [ tuningFrequency ] = adjustTuningFrequency( tuningFrequency, imSize )
       tuningFrequency = tuningFrequency-1;
       assert(all(size(tuningFrequency) ==[1,2]), 'Tuning Frequency is the wrong size');
       Ysz = imSize(1);
       Xsz = imSize(2);       
       tuningFrequency(1) = mod(pi+2*pi*(tuningFrequency(1)./Xsz), 2*pi)-pi;
       tuningFrequency(2) = mod(pi+2*pi*(tuningFrequency(2)./Ysz), 2*pi)-pi;

end

