% NOISEMODEL(SIGNALDEPENDENTNOISEMODEL, VARIANCE, BINCENTERS)
% 
% Represents signal dependent and constant variance noise models. If
% SIGNALDEPENDENTNOISEMODEL is false (i.e. using a constant noise model), 
% BINCENTERS is not required. Otherwise, VARIANCE must be a array
% of variances that is the same size as BINCENTERS. These two arrays are 
% linearly interpolated to represent the noise level function [Liu et al.
% 2006] with BINCENTERS representing the x-axis and VARIANCE the y-axis.
%
% Author(s): Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: August 2016
classdef NoiseModel
    
    properties
        signalDependentNoiseModel
        variance
        binCenters
    end
    
    methods
        function nm = NoiseModel(signalDependentNoiseModel, variance, binCenters)
            nm.signalDependentNoiseModel = signalDependentNoiseModel;
            nm.variance = variance;
            if (nm.signalDependentNoiseModel)
               nm.binCenters = binCenters;
               if (numel(nm.binCenters) ~= numel(nm.variance))
                   error('Number of elements in variance array must be the same as in binCenters')
               end
            end
        end                        
    end
    
    methods (Static)
        
        % Constant variance noise model
        function nm = GetConstantNoiseModel(variance)
            nm = NoiseModel(false, variance);
        end
                
        % Get pre-measured signal-dependent noise model from
        % Point Grey Grasshopper 3 Mono camera (GS3-U3-23S6M-C)
        % Estimated with unit gain applied to the image (0 dB of gain)
        function nm = GetPtGreyGS323S6MNoiseModel()
            M = load('PtGreyGS3-U3-23S5M-UnitGain.mat');
            nm = NoiseModel(true, M.variance, M.binCenters);            
        end
        
        
        
    end
    
end

