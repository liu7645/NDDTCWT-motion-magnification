function [ hL, hH, bL, bH, t, directL, directH ] = FilterTaps()
% Returns the lowpass and highpass filters specified in the supplementary
% materials of "Riesz Pyramid for Fast Phase-Based Video Magnification"
%
% Copyright Neal Wadhwa, August 2014
%
% Part of the Supplementary Material to:
%
% Riesz Pyramids for Fast Phase-Based Video Magnification
% Neal Wadhwa, Michael Rubinstein, Fredo Durand and William T. Freeman
% Computational Photography (ICCP), 2014 IEEE International Conference on
%
% hL and hH are the one dimenionsal filters designed by our optimization
% bL and bH are the corresponding Chebysheve polynomials
% t is the 3x3 McClellan transform matrix
% directL and directH are the direct forms of the 2d filters



hL =  [-0.0209 -0.0219 0.0900 0.2723 0.3611 0.2723 0.09 -0.0219 -0.0209];
hH =  [0.0099 0.0492 0.1230 0.2020 -0.7633 0.2020 0.1230 0.0492 0.0099];

% These are computed using Chebyshev polynomials, see filterToChebyCoeff
% for more details
bL = filterToChebyCoeff(hL);
bH = filterToChebyCoeff(hH);

% McClellan Transform
t = [1/8 1/4 1/8; 1/4 -1/2 1/4; 1/8 1/4 1/8];


directL = filterTo2D(bL, t);
directH = filterTo2D(bH, t);


end

%% 
% Returns the Chebyshev polynomial coefficients corresponding to a 
% symmetric 1D filter
function chebyshevPolyCoefficients = filterToChebyCoeff(taps)
    % taps should be an odd symmetric filter
    M = numel(taps);
    N = (M+1)/2; % Number of unique entries
    
    % Compute frequency response
    % g(1) + g(2)*cos(\omega) + g(3) \cos(2\omega) + ...
    g(1) = taps(N);
    g(2:N) = taps(N+1:end)*2;
    
    % Only need five polynomials for our filters
    ChebyshevPolynomial{1} = [0, 0, 0, 0, 1];
    ChebyshevPolynomial{2} = [0, 0, 0, 1, 0];
    ChebyshevPolynomial{3} = [0, 0, 2, 0, -1];
    ChebyshevPolynomial{4} = [0, 4, 0, -3, 0];
    ChebyshevPolynomial{5} = [8, 0, -8, 0, 1];
    
    
    % Now, convert frequency response to polynomials form
    % b(1) + b(2)\cos(\omega) + b(3) \cos(\omega)^2 + ...
    b = zeros(1,N);
    for k = 1:N
       p = ChebyshevPolynomial{k};       
       b = b + g(k)*p;
    end
    chebyshevPolyCoefficients = fliplr(b);


end


function impulseResponse = filterTo2D(chebyshevPolyCoefficients, mcClellanTransform)
    ctr = numel(chebyshevPolyCoefficients);
    N = 2*ctr-1;
    
    % Initial an impulse and then filter it
    X = zeros(N,N);
    X(ctr, ctr)= 1;
    
    
    Y(:,:,1) = X;
    for k = 2:numel(chebyshevPolyCoefficients);
        % Filter delta function repeatedly with the McClellan transform
        % Size of X is chosen so boundary conditions don't matter
        Y(:,:,k) = conv2(Y(:,:,k-1), mcClellanTransform,'same'); 
    end
    % Take a linear combination of these to get the full 2D response
    chebyshevPolyCoefficients = reshape(chebyshevPolyCoefficients, [1 1 numel(chebyshevPolyCoefficients)]);    
    impulseResponse = sum(bsxfun(@times, Y, chebyshevPolyCoefficients),3);
    
end