function reconstructed = reconNewPyr( pyr )
% Collapases a multi-scale pyramid of and returns the reconstructed image.
% pyr is a column vector, in which each level of the pyramid is
% concatenated, pind is the size of each level. 
%
% Copyright, Neal Wadhwa, August 2014
%

% Get the filter taps
% Because we won't be simultaneously lowpassing/highpassing anything and 
% most of the computational savings comes from the simultaneous application 
% of the filters, we use the direct form of the filters rather the 
% McClellan transform form
[~, ~, ~, ~, ~, directL, directH] = FilterTaps();
directL = 2*directL; % To make up for the energy lost during downsampling


nLevels = numel(pyr);
lo = pyr{nLevels};
for k = nLevels:-1:2
    upsz = size(pyr{k-1});
    % Upsample the lowest level
    lowest = zeros(upsz, 'single');
    lowest(1:2:end,1:2:end) = lo;
    % Lowpass it with reflective boundary conditions
    lowest = [lowest(5:-1:2,5:-1:2) lowest(5:-1:2,:) lowest(5:-1:2, end-1:-1:end-4); lowest(:,5:-1:2) lowest lowest(:,end-1:-1:end-4); lowest(end-1:-1:end-4,5:-1:2) lowest(end-1:-1:end-4,:) lowest(end-1:-1:end-4,end-1:-1:end-4)];
    lowest = conv2(lowest, directL, 'valid');
    % Get the next level    
    nextLevel = pyr{k-1};
    nextLevel = [nextLevel(5:-1:2,5:-1:2) nextLevel(5:-1:2,:) nextLevel(5:-1:2, end-1:-1:end-4); nextLevel(:,5:-1:2) nextLevel nextLevel(:,end-1:-1:end-4); nextLevel(end-1:-1:end-4,5:-1:2) nextLevel(end-1:-1:end-4,:) nextLevel(end-1:-1:end-4,end-1:-1:end-4)];
    % Highpass the level and add it to lowest level to form a new lowest
    % level
    lowest = lowest + conv2(nextLevel, directH, 'valid');
    lo = lowest;
end
reconstructed = lo;
end

