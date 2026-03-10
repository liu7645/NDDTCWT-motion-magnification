function [ delta ] = DisplacementFromMean( delta, fl, fh )
%DISPLACEMENTFROMMEAN Summary of this function goes here
%   Detailed explanation goes here
    delta = bsxfun(@minus, delta, mean(delta, 3));

end

