classdef MyVideoReader < handle
% A wrapper class for MATLAB's mmreader (now VideoReader) for enhanced
% functionality. For example, also supports image sequences.
% 
% Michael Rubinstein, MIT 2010
% Modified by Neal Wadhwa, MIT 2016

properties (Constant)
    % video type enum
    VID_FILE = 0 % video file
    VID_SEQ = 1 % image sequence
    VID_BLOCK = 2; % video block

    VIDOBJ_MMREADER = 0 % for older MATLAB versions
    VIDOBJ_VIDEOREADER = 1
    
    DEFAULTFRAMERATE = 25;
end


properties (GetAccess = public, SetAccess = immutable)
    width
    height
    length
    
    
    nChannels
    path
    name
    
    FrameRate
    Height
    Width
    Name
    NumberOfFrames
    
end

properties (SetAccess = private)
    vidType
    frames
    format
    vidObj
    vidObjType
end

properties
   flip; 
end

methods
    function o = MyVideoReader(filename)
        if (~isstr(filename))
            o.vidObj = filename;
            [o.height,o.width, o.nChannels, o.length] = size(o.vidObj);
            o.vidType = o.VID_BLOCK;
            o.FrameRate = o.DEFAULTFRAMERATE;
            o.name = 'NoName';
            
        else
            [o.path,o.name] = fileparts(filename);
            o.frames = [];
            o.vidObj = [];

            %--- Image sequence
            if isdir(filename) 
                o.frames = imdir(filename);
                ref = imread(fullfile(filename,o.frames(1).name));
                [o.height,o.width,o.nChannels] = size(ref);
                o.length = length(o.frames);
                [~,~,format] = fileparts(o.frames(1).name);
                o.format = format(2:end);
                o.vidType = o.VID_SEQ;
                o.FrameRate = o.DEFAULTFRAMERATE;
                
            %--- Video file
            else 
                if exist('VideoReader','class')==8
                    o.vidObj = VideoReader(filename);
                    o.width = o.vidObj.Width; 
                    o.height = o.vidObj.Height;
                    o.nChannels = o.vidObj.BitsPerPixel/8;
                    o.length = o.vidObj.NumberOfFrames;
                    o.vidObjType = o.VIDOBJ_VIDEOREADER;
                    o.FrameRate = o.vidObj.FrameRate;
                else % backward compatibility. Assumes mmreader exists...
                    o.vidObj = mmreader(filename);
                    o.width = get(o.vidObj,'Width');
                    o.height = get(o.vidObj,'Height');
                    o.length = get(o.vidObj,'NumberOfFrames');
                    o.format = get(o.vidObj,'VideoFormat');
                    o.vidObjType = o.VIDOBJ_MMREADER;
                end
                o.vidType = o.VID_FILE;
            end
        end
        o.Height = o.height;
        o.Width = o.width;
        o.Name = o.name;
        o.NumberOfFrames = o.length;
    end
    
    function delete(o)
        o.close;
    end

    function close(o)
        if o.vidType==o.VID_FILE
            clear o.vidObj;
        elseif o.vidType==o.VID_SEQ
            % TODO
        end
    end

    function [height,width,length,nChannels] = dimensions(o)
        height = o.height;
        width = o.width;
        length = o.length;
        nChannels = o.nChannels;
    end

    function res = resolution(o)
        res = [o.height,o.width];
    end
    
    function fr = frameRate(o)
        if o.vidType==o.VID_FILE
            fr = o.vidObj.FrameRate;
        else
            fr = [];
        end
    end

    % Fetchs frame t from the video
    function frame = frame(o,t)
        frame = [];
        if t<1 | t>o.length
            return;
        end
        if o.vidType==o.VID_FILE
            frame = read(o.vidObj,t);
        elseif o.vidType==o.VID_SEQ
            frame = imread(fullfile(o.path,o.name,o.frames(t).name));
        elseif (o.vidType == o.VID_BLOCK)
            frame = o.vidObj(:,:,:,t);
        end
        if (o.flip)
           frame = flipud(frame); 
        end
    end

    function frame = read(o, t)
        if (nargin == 1)
           frame = o.load();           
        elseif (numel(t) == 1)
            frame = o.frame(t);
        else
            for k = t(1):t(2)
               frame(:,:,:,k) = o.frame(k); 
            end
        end
        if (size(frame,3) == 1)
           frame = repmat(frame,[ 1 1 3 1]);  
        end
    end
    
    function name = frameName(o,t)
        name = [];
        if t<1 | t>o.length
            name = []; return;
        end
        if o.vidType==o.VID_FILE
            name = sprintf('%s_%05d',o.name,t);
        elseif o.vidType==o.VID_SEQ
            [tmp,name] = fileparts(o.frames(t).name);
        end
        
        
    end

    % Loads the video sequence to memory as a WxHxCxT matrix
    %  frames (optional) - vector of frames to load
    %  grayscale (optional) - load video in grayscale?
    function seq = load(o,frames,grayscale)
        if nargin<2
            frames = 1:o.length;
        end
        if nargin<3
            grayscale = 0;
        end
        
        nChannels = o.nChannels;
        if (grayscale), nChannels = 1; end

        % A quick memory check
        
        
       

        seq = zeros(o.height,o.width,o.nChannels,o.length,class(o.frame(frames(1))));
        for i=1:o.length
            im = o.frame(frames(i));
            if grayscale, im = rgb2gray(im); end
            seq(:,:,:,i) = im;
        end
    end


end % methods


end