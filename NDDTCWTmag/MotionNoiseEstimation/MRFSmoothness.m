% INTERPOLATEDMOTION = MRFSMOOTHNESS(MOTION, MOTIONCOVARIANCE, SMOOTHNESSTERM)
%
% Interpolates motions into noisy regions by assuming that adjacent pixels
% should have similar motions. Takes as input the estimated motions and
% their covariances. Then, minimizes the following objective function
% \sum_{over all pixels} (INTERPOLATEDMOTION - MOTION)' *
% inv(MOTIONCOVARIANCE) * (INTERPOLATEDMOTION - MOTION) + SMOOTHNESSTERM * 
% (MOTION - MOTIONNEIGHBORS)^2
% The first term is the data term, instructing the interpolated motions to
% be close to the true motions if they are reliable, while the second
% instructs the interpolated motions to be close to their neighbors.
% This objective is quadratic and is minimized by taking the derivative and
% solving for the point where the derivative is zero.
% The objective assumes the motions are Gaussian, which is only true at
% corners and in the direction perpendicular to edges. To mitigate this
% problem, the covariance should be increased to a large number at other
% points and directions.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function [ interpolatedMotion ] = MRFSmoothness(motion, motionCovariance, smoothnessTerm )
    % minimize (flow - smooth_flow)^2/lV + (
    % Laplacian matrix
    height = size(motion,1);
    width = size(motion,2);
    flatten = @(x) x(:);
    
    laplacianMatrix = getLaplacian(height, width);
    
    denominator = max(motionCovariance(:,:,1,1).*motionCovariance(:,:,2,2)-motionCovariance(:,:,1,2).^2,0) + 1e-15;
    D1 = 2*[flatten(motionCovariance(:,:,2,2))./denominator(:); flatten(motionCovariance(:,:,1,1))./denominator(:)];
    D2 = -2*flatten(motionCovariance(:,:,1,2))./denominator(:);
    D2 = [D2; D2];
    A = smoothnessTerm*laplacianMatrix + spdiags(D1,0,2*height*width, 2*height*width);
    A = A + spdiags(D2,height*width,2*height*width,2*height*width);
    A = A + spdiags(D2,-height*width,2*height*width,2*height*width);
    interpolatedMotion = zeros(height,width,2,size(motion,4),'single');
    for k = 1:size(motion,4)
        fprintf('On frame %d/%d\n', k, size(motion,4));
        horizontalMotion = motion(:,:,1,k);
        verticalMotion = motion(:,:,2,k);
        b = [(2*flatten(motionCovariance(:,:,2,2)).*horizontalMotion(:)-2*flatten(motionCovariance(:,:,1,2)).*verticalMotion(:))./denominator(:);
             (2*flatten(motionCovariance(:,:,1,1)).*verticalMotion(:)-2*flatten(motionCovariance(:,:,1,2)).*horizontalMotion(:))./denominator(:)];
        interpolatedMotionCurrentFrame = A\double(b);
        interpolatedMotion(:,:,:,k) = reshape(interpolatedMotionCurrentFrame, [height, width,2]);
    end
    

end

% LAPLACIANMATRIX = GETLAPLACIAN(HEIGHT, WIDTH)
%
% Construct a sparse Laplacian matrix for use in MRFSmoothness. This matrix
% is of dimension 2*HEIGHT*WIDTH by 2*HEIGHT*WIDTH
function laplacianMatrix = getLaplacian(height, width)
    count = 1;
    II = zeros(height*width*5,1);
    JJ = zeros(height*width*5,1);
    SS = zeros(height*width*5,1);
    for x = 1:width
        for y = 1:height
            total = 0;
            s = ind(y,x);
            if (y >= 2)
                s1 = ind(y-1,x);
                II(count) = s;
                JJ(count) = s1;
                SS(count) = -1;
                count = count + 1;
                total = total + 1;
            end
            
            if (y < height)
                s2 = ind(y+1,x);
                II(count) = s;
                JJ(count) = s2;
                SS(count) = -1;
                count = count + 1;
                total = total + 1;
            end
            
            if (x > 1)
                s3 = ind(y,x-1);
                II(count) = s;
                JJ(count) = s3;
                SS(count) = -1;
                count = count + 1;
                total = total + 1;
            end
            
            if (x < width)
                s4 = ind(y,x+1);
                II(count) = s;
                JJ(count) = s4;
                SS(count) = -1;
                count = count + 1;
                total = total + 1;
            end
            II(count) = s;
            JJ(count) = s;
            SS(count) = total;
            count = count + 1;                        
        end
    end
    count = count -1;
    II = II(1:count);
    JJ = JJ(1:count);
    SS = SS(1:count);
    laplacianMatrix = sparse(II,JJ,SS, height*width, height*width);
    laplacianMatrix = [laplacianMatrix sparse(height*width, height*width); sparse(height*width, height*width) laplacianMatrix];
    function idx =ind(i,j)
       idx = sub2ind([height,width],i,j); 
    end
end

