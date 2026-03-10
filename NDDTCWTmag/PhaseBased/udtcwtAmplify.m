% PHASEAMPLIFY(VIDFILE, MAGPHASE, FL, FH, FS, OUTDIR, VARARGIN) 
% 
% Takes input VIDFILE and motion magnifies the motions that are within a
% passband of FL to FH Hz by MAGPHASE times. FS is the videos sampling rate
% and OUTDIR is the output directory. 
%
% Optional arguments:
% attenuateOtherFrequencies (false)
%   - Whether to attenuate frequencies in the stopband  
% pyrType                   ('halfOctave')
%   - Spatial representation to use (see paper)
% sigma                     (0)            
%   - Amount of spatial smoothing (in px) to apply to phases 
% temporalFilter            (FIRWindowBP) 
%   - What temporal filter to use
% 

function outName = udtcwtAmplify(vr, magPhase , fl, fh,fs, outDir, varargin)

 %   global timeTic;
    %global test;
    %% Read Video
    writeTag = vr.Name;
    FrameRate = vr.FrameRate;    
%     nF = vr.NumberOfFrames;

 %   timeTic=zeros(1,6);
%   tic;
    fprintf('Begin MyFunction\n');    
%     if(test==1)        
%         [bl,br,bu,bd,maxFuzzyOut]=MyFunction(vr,vid);        
%         vid = vid(bu:bd,bl:br,:,:);
%     end  
%     timeTic(2)=toc;
%     fprintf('End MyFunction\n');
    %% Parse Input
    p = inputParser();

    defaultAttenuateOtherFrequencies = false; %If true, use reference frame phases
    pyrTypes = {'octave', 'halfOctave', 'smoothHalfOctave', 'quarterOctave'}; 
    defaultGetFilenameOnly = false;
    checkPyrType = @(x) find(ismember(x, pyrTypes));
    defaultPyrType = 'octave';
    defaultSigma = 0;
    defaultTemporalFilter = @FIRWindowBP;
    defaultScale = 1;
    defaultFrames = [1, 400];
    
    addOptional(p, 'attenuateOtherFreq', defaultAttenuateOtherFrequencies, @islogical);
    addOptional(p, 'getFilenameOnly', defaultGetFilenameOnly);
    addOptional(p, 'pyrType', defaultPyrType, checkPyrType);
    addOptional(p,'sigma', defaultSigma, @isnumeric);   
    addOptional(p, 'temporalFilter', defaultTemporalFilter);
    addOptional(p, 'scaleVideo', defaultScale);
    addOptional(p, 'useFrames', defaultFrames);
    
    parse(p, varargin{:});

    refFrame = 1;
    attenuateOtherFreq = p.Results.attenuateOtherFreq;
    pyrType            = p.Results.pyrType;
    sigma              = p.Results.sigma;
    temporalFilter     = p.Results.temporalFilter;
    scaleVideo         = p.Results.scaleVideo;
    frames             = p.Results.useFrames;
    getFilenameOnly    = p.Results.getFilenameOnly;
    
   
    %% Compute spatial filters        
    frame = imresize(vr.read(1), scaleVideo);
    [h, w, ~] = size(frame);
    [Faf, Fsf] = NDAntonB2; %(Must use ND filters for both)
   [af, sf] = NDdualfilt1;
    fprintf('Computing spatial filters\n');
    ht = 3;%层数
    switch pyrType
        case 'octave'
            filters = getFilters([h w], 2.^[0:-1:-ht], 4);%返回金字塔，图像尺寸，过滤器边界，指定方向
            repString = 'octave';
            fprintf('Using octave bandwidth pyramid\n');        
        case 'halfOctave'            
            filters = getFilters([h w], 2.^[0:-0.5:-ht], 8,'twidth', 0.75);
            repString = 'halfOctave';
            fprintf('Using half octave bandwidth pyramid\n'); 
        case 'smoothHalfOctave'
            filters = getFiltersSmoothWindow([h w], 8, 'filtersPerOctave', 2);           
            repString = 'smoothHalfOctave';
            fprintf('Using half octave pyramid with smooth window.\n');
        case 'quarterOctave'
            filters = getFiltersSmoothWindow([h w], 8, 'filtersPerOctave', 4);
            repString = 'quarterOctave';
            fprintf('Using quarter octave pyramid.\n');
        otherwise 
            error('Invalid Filter Types');
    end
    %% Return if only filename is needed
    outName = fullfile(outDir, sprintf('%s-%s-band%0.2f-%0.2f-sr%d-alpha%d-mp%d-sigma%d-scale%0.2f-frames%d-%d-%s.avi', writeTag, func2str(temporalFilter), fl, fh,fs, magPhase, attenuateOtherFreq, sigma, scaleVideo, frames(1), frames(2), repString));
    if (getFilenameOnly)
        return
    end

    %% Read Video
    vid = vr.read([frames(1), frames(2)]);   %4-D uint8,一个4维数组,里面的元素是uint8型; frames(1)取矩阵第一个元素1，frames(2)取矩阵第二个元素347，即从1到347帧 
    %vid1 = rgb2ntsc(im2single(vid(:,:,:,1)));
    
    [h, w, nC, nF] = size(vid);
    if (scaleVideo~= 1)
        [h,w] = size(imresize(vid(:,:,1,1), scaleVideo));
    end


    %% Initialization of motion magnified luma component
%     magnifiedLumaFFT = zeros(h,w,nF,'single');
    
    fprintf('Begin compute phase differences\n');
    %% First compute phase differences from reference frame
    fprintf('Moving video to Fourier domain\n');

    for k = 1:nF
        originalFrame = rgb2ntsc(im2single(vid(:,:,:,k)));%oF包含亮度 色调 饱和度      
        tVid(:,:,k) = imresize(originalFrame(:,:,1), [h w]);% imresize尺寸缩放，缩了一半，[h w]指定目标图像的高度和宽度，这里只取亮度通道
        %vidFFT(:,:,k) = unwrap_phase(vidFFT(:,:,k));
    end
    clear vid;
%    timeTic(4)=toc;
    fprintf('Begin Compute phases of level\n');
    for k = 1:nF
        k
        
        pyr = NDxWav2DMEX(tVid(:,:,k) , (ht+1), Faf, af, 1);   %实部是余弦滤波器所得 ，虚部是正弦滤波器所得  线性空间滤波是空间卷积，基于相位是频域与特定滤波器（模拟高斯的）相乘      
        %因为滤波器在空域里实部是偶函数，虚部是奇函数，那么空域信号经过偶滤波器后就是实数，经过奇滤波器就是虚部，图像乘以滤波器的反傅里叶变换，是空域里的傅里叶级数，傅里叶级数的系数是复数，系数实部相当于频域里的实部，系数虚部相当于频域里的虚部。
        %curLevelFrame = reconLevel(pyrRef, level);
        pyramppha{k} = pyr;
        
        for level = 1:(ht+1)
            for nothinggood = 1:2
                for originala = 1:3
                    pyr1 = 1i*pyr{level}{2}{nothinggood}{originala}+pyr{level}{1}{nothinggood}{originala};
                    pyramppha{k}{level}{1}{nothinggood}{originala} = abs(pyr1);
                    pyramppha{k}{level}{2}{nothinggood}{originala} = angle(pyr1);

                end
            end
        end

    end
    S = whos('pyramppha');
    peak_memory_gb = S.bytes / (1024^3);
    fprintf('NDDTCWT Cell Array 峰值内存占用: %.2f GB\n', peak_memory_gb);                
        fprintf('Processing level %d of %d\n', level, ht);
        %pyrCurrent1 = zeros(size(pyrRef,1), size(pyrRef,2) ,nF,'single');

        %% Apply magnification

        fprintf('Applying magnification\n');
        delta = zeros(h, w ,nF,'double');
%         amp = zeros(h, w ,'single');
%         pha = zeros(h, w ,'single');
    for level = 1:(ht+1)
  
        for nothinggood = 1:2
            for originala = 1:3
                for k = 1:nF
%                     delta(:,:,k) = single(mod(pi+pyramppha{k}{level}{2}{nothinggood}{originala}...
%                         - pyramppha{1}{level}{2}{nothinggood}{originala},2*pi)-pi);
                        delta(:,:,k) = pyramppha{k}{level}{2}{nothinggood}{originala}...
                         - pyramppha{1}{level}{2}{nothinggood}{originala};
                    %delta(:,:,k) = Unwrap_TIE_DCT_Iter(delta(:,:,k));
                    
                    
                end
                delta = temporalFilter(delta, fl/fs,fh/fs); 
                for k = 2:nF
%                     if (sigma~= 0)
% 		               phaseOfFrame = AmplitudeWeightedBlur(delta(:,:,k), pyramppha{k}{level}{1}{nothinggood}{originala}+eps, sigma);  
% 	                end
	                pha(:,:) = pyramppha{k}{level}{2}{nothinggood}{originala} + delta(:,:,k) * magPhase;
                    amp(:,:) = pyramppha{k}{level}{1}{nothinggood}{originala};
                    pyramppha{k}{level}{1}{nothinggood}{originala} = amp .* cos(pha);
                    pyramppha{k}{level}{2}{nothinggood}{originala} = amp .* sin(pha);
                    
                end

            end
        end

    end
    clear tVid delta;
for level =1:(ht+1)
        for nothinggood = 1:2
            for originala = 1:3
         
	                pha(:,:) = pyramppha{1}{level}{2}{nothinggood}{originala};
                    amp(:,:) = pyramppha{1}{level}{1}{nothinggood}{originala};
                    pyramppha{1}{level}{1}{nothinggood}{originala} = amp .* cos(pha);
                    pyramppha{1}{level}{2}{nothinggood}{originala} = amp .* sin(pha);
            end

    end
end


   
    %% Add unmolested lowpass residual

    fprintf('Begin Put frame in output\n');
    vid = vr.read([frames]);
    res = zeros(h,w,nC,nF,'uint8');
    for k = 1:nF
        k
        ht
        magnifiedLuma = NDixWav2DMEX(pyramppha{k}, ht+1, Fsf, sf, 1);
        k
        outFrame(:,:,1) = magnifiedLuma;
        originalFrame = rgb2ntsc(im2single(vid(:,:,:,k)));    
        originalFrame = imresize(originalFrame, [h, w]);
        outFrame(:,:,2:3) = originalFrame(:,:,2:3);
        outFrame = ntsc2rgb(outFrame);        
        %% Put frame in output
        res(:,:,:,k) = im2uint8(outFrame);
    end
%    timeTic(6)=toc;
    fprintf('Begin writeVideo\n');

%     sf1 = SpatialFrequency(zuihousfliang)
    writeVideo(res, FrameRate, outName);   
end

function phase = un2wrap(phase)
    a=phase;                                             %将包裹相位ph赋值给a
    %下面开始进行最小二乘解包裹运算
    [M,N]=size(a);                                    %计算二维包裹相位的大小（行、列数）
    dx=zeros(M,N);dy=zeros(M,N);                      %预设包裹相位沿x方向和y方向的梯度
    m=1:M-1; 
    dx(m,:)=a(m+1,:)-a(m,:);                          %计算包裹相位沿x方向的梯度
    dx=dx-pi*round(dx/pi);                            %去除梯度中的跳跃
    n=1:N-1;
    dy(:,n)=a(:,n+1)-a(:,n);                          %计算包裹相位沿y方向的梯度
    dy=dy-pi*round(dy/pi);                            %去除梯度中的跳跃
    p=zeros(M,N);p1=zeros(M,N);p2=zeros(M,N); %为计算ρnm作准备
    m=2:M;
    p1(m,:)=dx(m,:)-dx(m-1,:);                        %计算Δgxnm-Δgx(n-1)m
    n=2:N;
    p2(:,n)=dy(:,n)-dy(:,n-1);                        %计算Δgynm–Δgyn(m-1)
    p=p1+p2;                                          %计算ρnm
    p(1,1)=dx(1,1)+dy(1,1);                           %计算ρnm
    n=2:N;
    p(1,n)=dx(1,n)+dy(1,n)-dy(1,n-1);                 %赋值Neumann边界条件
    m=2:M;
    p(m,1)=dx(m,1)-dx(m-1,1)+dy(m,1);
    pp=dct2(p)+eps;                                   %计算ρnm的DCT
    fi=zeros(M,N);
    for m=1:M                                         %计算Φnm在DCT域的精确解
       for n=1:N  
          fi(m,n)=pp(m,n)/(2*cos(pi*(m-1)/M)+2*cos(pi*(n-1)/N)-4+eps);
       end
    end
    fi(1,1)=pp(1,1);                                  %赋值DCT域的Φ11
    phase=idct2(fi);
end