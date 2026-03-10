function pyr = buildNewPyr( im )
% Returns a multi-scale pyramid of im. pyr is the pyramid concatenated as a
% column vector while pind is the size of each level. im is expected to be
% a grayscale two dimenionsal image in either single floating
% point precision.
%
% Copyright, Neal Wadhwa, August 2014
%
% Part of the Supplementary Material to:
%
% Riesz Pyramids for Fast Phase-Based Video Magnification
% Neal Wadhwa, Michael Rubinstein, Fredo Durand and William T. Freeman
% Computational Photography (ICCP), 2014 IEEE International Conference on

% Get the filter taps
[~, ~, bL, bH, t, ~, ~] = FilterTaps();
bL = reshape(bL, [1 1 numel(bL)]);
bL = 2 * bL; % To make up for the energy lost during downsampling
bH = reshape(bH, [1 1 numel(bH)]);

im_sz = size(im);

pyrIDX = 1;
while (not(any(im_sz<10))) % Stop building the pyramid when the image is too small
    Y = zeros(im_sz(1), im_sz(2), numel(bL),'single'); % 
    Y(:,:,1) = im; 
    % We apply the McClellan transform repeated to the image
    for k = 2:numel(bL)
       previousFiltered = Y(:,:,k-1);
       % Reflective boundary conditions
       previousFiltered = [previousFiltered(2,2), previousFiltered(2,:), previousFiltered(2,end-1); previousFiltered(:,2) previousFiltered previousFiltered(:,end-1); previousFiltered(end-1,2), previousFiltered(end-1,:) previousFiltered(end-1,end-1)];            
       Y(:,:,k) = conv2(previousFiltered, t, 'valid');
    end
    
    % Use Y to compute lo and highpass filtered image    
    lopassedIm = sum(bsxfun(@times, Y, bL),3);
    hipassedIm = sum(bsxfun(@times, Y, bH),3);
    
    % Add highpassed image to the pyramid
    pyr{pyrIDX} = hipassedIm;
    pyrIDX = pyrIDX + 1;
    
    % Downsample lowpassed image
    lopassedIm = lopassedIm(1:2:end,1:2:end);
    
    % Recurse on the lowpassed image
    im_sz = size(lopassedIm);
    im = lopassedIm;
end
% Add a residual level for the remaining low frequencies
pyr{pyrIDX} =  im;

end

