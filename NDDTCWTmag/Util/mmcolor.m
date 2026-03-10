% Metamaterial colorbar; matches colorbar produced by Abaqus
function [ output ] = mmcolor( N )
    if (nargin < 1)
        N = 64;
    end
    output = zeros(N,3,'double');
    X = linspace(0,1,N)';
    % Red
    output(:,1) = clip((-4*X+2), 0, 1);
    % Green
    output(:,2) = (clip(4*X,0,1).*clip(-4*X+4,0,1));
    % Blue
    output(:,3) = clip(4*X-2,0,1);
    
    output=  fliplr(output);
end

