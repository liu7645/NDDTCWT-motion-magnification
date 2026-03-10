classdef FileVR
    % Like videoreader class, but takes as input a three dimensional or
    % four dimensional arrays whose last dimension is time. Treats each
    % time instance as frame of a video.
    %
    % Neal Wadhwa, MIT 2016   
    properties
        vid;
        Name;
        FrameRate =25;
    end
    
    methods
        function VR = FileVR(vid, name)
            if (ndims(vid) == 4)
                VR.vid = vid;
            else
               VR.vid = permute(vid, [1 2 4 3]); 
            end
            if (nargin ==2)
               VR.Name = name;
            else
                VR.Name = 'NoNameGiven';
            end
        end
        
        function out = read(VR,frameRange)
            if (exist('frameRange', 'var'))
                frameRange = ceil(frameRange);
                if (numel(frameRange) == 1)                
                    if (or(frameRange <1, frameRange > VR.NumberOfFrames))
                        error('Invalid value of frame range.');
                    end
                    out = VR.vid(:,:,:,frameRange);
                elseif (numel(frameRange) == 2)                
                    if (frameRange(2) <= frameRange(1))
                        error('Second element of frame range must be larger than first.\n');
                    end
                    out = VR.vid(:,:,:,frameRange(1):frameRange(2));
                else 
                    error('Argument must be a single number or a two element array\n');
                end
            else
                out = VR.vid;
            end
        end

            
            
        function out = NumberOfFrames(VR)
           out = size(VR.vid,4); 
        end
    end
    
    
end

