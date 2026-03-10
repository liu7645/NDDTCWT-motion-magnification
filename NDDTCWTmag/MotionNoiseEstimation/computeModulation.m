% MODULATION = COMPUTEMODULATION(TUNINGFREQUENCY, IMSIZE) 
% 
% Computes a 2D sinusoidal MODULATION with frequency given by
% TUNINGFREQUENCY and image size given by IMSIZE. TUNINGFREQUENCY should be
% a two element normalized frequency with each element in the range 
% [-pi, pi].
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function modulation = computeModulation( tuningFrequency, imSize)
    
   % Compute modulation based on tuning Frequency   
   Ysz = imSize(1);
   Xsz = imSize(2);
   xx = (0:(Xsz - 1));
   yy = (0:(Ysz - 1));
   [xx, yy] = meshgrid(xx, yy);
    modulation = exp(1i*(xx*tuningFrequency(1) + yy * tuningFrequency(2)));

end

