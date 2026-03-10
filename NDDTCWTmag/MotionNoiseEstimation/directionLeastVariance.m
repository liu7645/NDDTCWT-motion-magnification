% [UNITVECTOR, LEASTVARIANCE, MOSTVARIANCE, NORMALVECTOR] = DIRECTIONLEASTVARIANCE (MOTIONCOVARIANCE)
%
% Rotates the motions at each pixel, so that the components are
% uncorrelated. The result is a direction of least variance (UNITVECTOR)
% with variance LEASTVARIANCE and the perpendicular direction
% (NORMALVECTOR) with MOSTVARIANCE. UNITVECTOR and NORMALVECTOR are given
% by complex number of unit magnitude.
%
% See InvertSpin.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function [ unitVector, leastVariance, mostVariance, normalVector ] = directionLeastVariance( motionCovariance )
    [height, width, ~] = size(motionCovariance);    
    unitVector = zeros(height,width);
    leastVariance = zeros(height,width);
    mostVariance = zeros(height,width);
    if (nargout == 4)
       normalVector = zeros(height,width); 
    end
    for m = 1:height
        for n = 1:width
            S = squeeze(motionCovariance(m, n, :, :));
            [V,D] = eig(S);
            D = diag(D);
            [~,I] = min(D);
            leastVariance(m,n) = D(I);
            mostVariance(m,n) = D(3-I);
            unitVector(m,n) = V(1,I)+1i*V(2,I);
            if (nargout ==4)
                normalVector(m,n) = V(1,3-I) + 1i*V(2,3-I);
            end
        end
    end


end

