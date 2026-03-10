function Bridgeudtcwt(dataDir, outDir, videofile)

 vr = VideoReader(fullfile(dataDir, videofile));

        sigma = 0;
    samplingRate = 30;
    temporalFilter = @Butterworth2ndOrder;
    nF =400;% vr.NumberOfFrames;
    frameRange = [1 nF]; 
    amplification = [400, 250];
    loCutoff = [1.6];
    hiCutoff = [1.8];

    for mode =  1:1
        
        % Motion amplification
        
        MotionAmplificationFunction = @(x) udtcwtAmplify(vr, amplification(mode), ...
            loCutoff(mode), hiCutoff(mode), samplingRate, outDir, ...
            'sigma', sigma, 'temporalFilter', temporalFilter, ...
            'useFrames', frameRange);
        
        requr = MotionAmplificationFunction(mode)
        
 
    end    
    
end

