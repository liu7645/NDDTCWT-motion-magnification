% Should produce identical results to buildLpyr in matlabPyrTools, but is
% much faster because it avoids the overhead of selecting the pyramid type.
% In addition, the output format is a cell array, with each element
% representing a level of the pyramid.
function pyr = buildLpyrFast( im  )

% Remove overhead from buildLpyr
im_sz = size(im);
filt1 =  [1 4 6 4 1]'/16*sqrt(2);
filt2 = filt1;

pyr = {};
count = 1;
while (not(any(im_sz<6)))
    
    %% Lowpass
   lo = rconv2fastX(im, filt1');
   lo = lo(:,1:2:end);
   int_sz = size(lo);
   lo = rconv2fastY(lo, filt1);
   lo2 = lo(1:2:end,:);
   
   
   %% Hipass
   hi = zeros(int_sz);
   hi(1:2:end,:) = lo2;
   hi = rconv2fastY(hi, filt2);
   hi2 = zeros(im_sz);
   hi2(:,1:2:end) = hi;
   hi2 = rconv2fastX(hi2, filt2');
   hi2 = im-hi2;
   
   
   pyr{count} = hi2;
   count = count + 1;
   
   
   im = lo2;
   im_sz = size(im); 
end
pyr{count} = im;


function c = rconv2fastX(large, small)
    clarge = [large(:, 3:-1:2), large, large(:,end-1:-1:end-2)];
    c = conv2(clarge, small,'valid');
    
    
function c = rconv2fastY(large, small)
    clarge = [large(3:-1:2,:); large; large(end-1:-1:end-2,:)];
    c = conv2(clarge, small,'valid');
    

