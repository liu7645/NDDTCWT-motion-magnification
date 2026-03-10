% MOTIONCOVARIANCE = INVERTSPIN(UNITVECTOR, LEASTVARIANCE, MOSTVARIANCE, NORMALVECTOR)
% 
% Undoes directionLeastVariance.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function [ motionCovariance ] = InvertSpin( unitVector, leastVariance, mostVariance, normalVector )
    motionCovariance(:,:,1,1) = real(unitVector).^2.*leastVariance + real(normalVector).^2.*mostVariance;
    motionCovariance(:,:,1,2) = real(unitVector).*imag(unitVector).*leastVariance + real(normalVector).*imag(normalVector).*mostVariance;
    motionCovariance(:,:,2,1) = motionCovariance(:, :, 1,2);
    motionCovariance(:,:,2,2) = imag(unitVector).^2.*leastVariance + imag(normalVector).^2.*mostVariance;
end

