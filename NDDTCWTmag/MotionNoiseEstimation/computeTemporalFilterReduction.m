% REDUCTION = COMPUTETEMPORALFILTERREDUCTION( TEMPORALFILTER, NF )
%
% Computes the REDUCTION in variance when a signal of independent Gaussians
% is filtered with TEMPORALFILTER. See Oppenheim and Schafer for details on
% why this is the correct formula. TEMPORALFILTER should process the third
% dimension of the input and NF is the number of elements in the signal
% being filtered.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
function reduction = computeTemporalFilterReduction( temporalFilter, nF )
    impulse = zeros(1,1,nF);
    impulse(1,1,fix(end/2)) = 1;
    impulseResponse = temporalFilter(impulse);
    reduction = sum(impulseResponse.*impulseResponse);
end

