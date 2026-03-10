function [ res ] = reconLpyrFast( pyr )
% Same as reconLpyr, but without overhead of parsing inputs. See
% buildLpyrFast for more details.
% Neal Wadhwa, Dec 2013
    
filt = sqrt(2)*[1,4,6,4,1]'/16;
nLevels = numel(pyr);
band = pyr{nLevels};
for k = nLevels:-1:2    
    int_sz = [size(pyr{k-1},1), size(pyr{k},2)];
    hi = zeros(int_sz);
    hi(1:2:end,:) = band;
    hi = rconv2fastY(hi, filt);
    res = zeros(size(pyr{k-1}));
    res(:,1:2:end) = hi;
    res = rconv2fastX(res, filt');
    band = pyr{k-1};
    band = res+ band;
    
end
res = band;

function c = rconv2fastX(large, small)
    clarge = [large(:, 3:-1:2), large, large(:,end-1:-1:end-2)];
    c = conv2(clarge, small,'valid');
    
    
function c = rconv2fastY(large, small)
    clarge = [large(3:-1:2,:); large; large(end-1:-1:end-2,:)];
    c = conv2(clarge, small,'valid');