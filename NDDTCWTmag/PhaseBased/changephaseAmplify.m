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

function outName = changephaseAmplify(vr, magPhase , fl, fh,fs, outDir, varargin)

 %   global timeTic;
    %global test;
    %% Read Video
    writeTag = vr.Name;
    FrameRate = vr.FrameRate;    
    nF = vr.NumberOfFrames;

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
    defaultFrames = [1, nF];
    
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
    
    fprintf('Computing spatial filters\n');
    ht = maxSCFpyrHt(zeros(h,w));%层数
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

    [croppedFilters, filtIDX] = getFilterIDX(filters);%coppedFIlters获取的是不同方向不同大小的空间卷积核，
                                                      %filtIDX放的是上面的卷积核的行索引组成的行向量和列索引组成得行向量
    %save('croppedFilters.mat','croppedFilters')
    %save('filtIDX.mat','filtIDX')
    %% Read Video
    vid = vr.read([frames(1), frames(2)]);   %4-D uint8,一个4维数组,里面的元素是uint8型; frames(1)取矩阵第一个元素1，frames(2)取矩阵第二个元素347，即从1到347帧 
    %vid1 = rgb2ntsc(im2single(vid(:,:,:,1)));
    
    [h, w, nC, nF] = size(vid);
    if (scaleVideo~= 1)
        [h,w] = size(imresize(vid(:,:,1,1), scaleVideo));
    end


    %% Initialization of motion magnified luma component
    magnifiedLumaFFT = zeros(h,w,nF,'single');
    
    buildLevel = @(im_dft, k) ifft2(ifftshift(croppedFilters{k}.* ...%只定义未调用 反傅里叶变换 时域卷积，频域乘积，用滤波器乘以对应索引的信号
        im_dft(filtIDX{k,1}, filtIDX{k,2})));%https://blog.csdn.net/myathappy/article/details/51344618 
    
    reconLevel = @(im_dft, k) 2*(croppedFilters{k}.*fftshift(fft2(im_dft)));
%    timeTic(3)=toc;
    fprintf('Begin compute phase differences\n');
    %% First compute phase differences from reference frame
    numLevels = numel(filters);  %返回元素个数      
    fprintf('Moving video to Fourier domain\n');
    vidFFT = zeros(h,w,nF,'single');
    
    for k = 1:nF
        originalFrame = rgb2ntsc(im2single(vid(:,:,:,k)));%oF包含亮度 色调 饱和度      
        tVid = imresize(originalFrame(:,:,1), [h w]);% imresize尺寸缩放，缩了一半，[h w]指定目标图像的高度和宽度，这里只取亮度通道
        vidFFT(:,:,k) = single(fftshift(fft2(tVid)));%使用快速傅里叶变换算法返回矩阵的二维傅里叶变换，这等同于计算 fft(fft(X).’).’（幅度谱平移去噪）
        %vidFFT(:,:,k) = unwrap_phase(vidFFT(:,:,k));
    end
%     zuihousf = rgb2ntsc(im2double(vid(:,:,:,k)))
%     zuihousfliang = zuihousf(:,:,1)
    clear vid;
%    timeTic(4)=toc;
    fprintf('Begin Compute phases of level\n');
    for level = 2:numLevels-1
        %% Compute phases of level
        % We assume that the video is mostly static
        pyrRef = buildLevel(vidFFT(:,:,refFrame), level);   %实部是余弦滤波器所得 ，虚部是正弦滤波器所得  线性空间滤波是空间卷积，基于相位是频域与特定滤波器（模拟高斯的）相乘      
        %因为滤波器在空域里实部是偶函数，虚部是奇函数，那么空域信号经过偶滤波器后就是实数，经过奇滤波器就是虚部，图像乘以滤波器的反傅里叶变换，是空域里的傅里叶级数，傅里叶级数的系数是复数，系数实部相当于频域里的实部，系数虚部相当于频域里的虚部。
        %curLevelFrame = reconLevel(pyrRef, level);
%         sfpyr = real(ifft2(ifftshift(curLevelFrame)))*255;
%         sfpyr = abs(sfpyr);
%         h1 = figure;
% 
%         imshow(sfpyr)
%         filenam =  fullfile(outDir, sprintf('%s.fig',num2str(level)))
%         savefig(h1,filenam)
%         close(h1);
        %sf(level) = SpatialFrequency(sfpyr)
        %sfpyr = im2uint8(sfpyr);
        %imwrite(sfpyr,level,'.jpg');
        
        %pyrRefPhaseOrig = pyrRef./abs(pyrRef);%归一化，将反傅里叶变换的结果归一化，使其模值为1
        pyrRefPhaseOrig = pyrRef;
        pyrRef = angle(pyrRef);        
        %pyrRef = Unwrap_TIE_DCT_Iter(pyrRef);
%         if magPhase == 5
%             pyrReffilenam =  fullfile(outDir, sprintf('pyrRef%s.mat',num2str(level)));
%             save(pyrReffilenam,'pyrRef');
%         end
        delta = zeros(size(pyrRef,1), size(pyrRef,2) ,nF,'single');
        fprintf('Processing level %d of %d\n', level, numLevels);
        %pyrCurrent1 = zeros(size(pyrRef,1), size(pyrRef,2) ,nF,'single');
        
        for frameIDX = 1:nF
            filterResponse = buildLevel(vidFFT(:,:,frameIDX), level);%frameIDX代表从第一帧图像开始， level代表从第二个滤波器开始
            
            pyrCurrent = angle(filterResponse);
%             if frameIDX ==  && magPhase == 5 
%                 pyrCurrentfilenam = fullfile(outDir, sprintf('pyrRef%sframe.mat',num2str(level)));
%                 save(pyrCurrentfilenam,'pyrCurrent');
%             end
            %pyrCurrent = Unwrap_TIE_DCT_Iter(pyrCurrent);
%             if frameIDX == 1 && magPhase == 5
%                 pyrCurrentunfilenam = fullfile(outDir, sprintf('pyrRef%sframeun.mat',num2str(level)));
%                 save(pyrCurrentunfilenam,'pyrCurrent');
%             end    
            delta(:,:,frameIDX) = single(pyrCurrent-pyrRef); 
            delta(:,:,frameIDX) = Unwrap_TIE_DCT_Iter(delta(:,:,frameIDX));
            %pyrCurrent1(:,:,frameIDX) = pyrCurrent;
            %delta(:,:,frameIDX) = single(mod(pi+pyrCurrent-pyrRef,2*pi)-pi);        %计算相位差   mod(a,b） a/b取余                 
            
        end
        
        
        %save('delta.mat','delta')
        %% Temporal Filtering
        fprintf('Bandpassing phases\n');
        
        delta = temporalFilter(delta, fl/fs,fh/fs); 
        %save('deltaFIR.mat','delta')
        if magPhase == 5
            deltafilenam = fullfile(outDir, sprintf('delta%s.mat',num2str(level)));
            save(deltafilenam,'delta');
        end

        %% Apply magnification
        magup = fix((level - 2) / 4);
        fprintf('Applying magnification\n');
        for frameIDX = 1:nF

            phaseOfFrame = delta(:,:,frameIDX);%去掉高低频通带
            originalLevel = buildLevel(vidFFT(:,:,frameIDX),level);%复值可操纵金字塔level层
            %% Amplitude Weighted Blur        
            if (sigma~= 0)
                phaseOfFrame = AmplitudeWeightedBlur(phaseOfFrame, abs(originalLevel)+eps, sigma);        
            end%eps=0  sigma是加权高斯函数的标准差 
            %magup = fix((level - 2) / 4);
            % Increase phase variation
            phaseOfFrame = magPhase *phaseOfFrame;  
            
            if (attenuateOtherFreq)%按模值大小缩放第一帧的实部虚部，因为该复值是某频率分量的系数，那么按第一帧来分配实部虚部的比例，即相位不变位置不变，最后相当于将与第一帧图像的相位差放大后叠加到第一帧上
                tempOrig = abs(originalLevel).*pyrRefPhaseOrig;%attenuateOtherFreq衰减其他频率(理论上某个空间频率的模式不变的，但是由于镜头的晃动以及噪声的影响会产生些微变化，attenuateOtherFreq=1会抑制这样的频率变化)
            else
                tempOrig = originalLevel;%将与第一帧图像的相位差放大后叠加到第n帧上
            end
            tempOrigabs = abs(tempOrig);
            tempOrigtheta = angle(tempOrig);
            %tempOrigtheta = Unwrap_TIE_DCT_Iter(tempOrigtheta);
            tempOrigtheta = tempOrigtheta + phaseOfFrame;
            if frameIDX == 5  && magPhase == 5 
                tempOrigtheta = fullfile(outDir, sprintf('tempOrigtheta%sframe.mat',num2str(level)));
                save(pyrCurrentfilenam,'pyrCurrent');
            end
            displace =fix(tempOrigtheta / pi);
            
            [row col] = size(tempOrigtheta);
            

            if magup == 0
                for i = 2: row-1
                    for j = 2: col - 1
                        fixtem = fix(tempOrigtheta(i,j) / pi)
                        if fixtem > 0 %&& col(i) - displace(row, col)>0
                            
                            suoyin1 = max(j - fixtem, 1)
                            suoyin2 = suoyin1 + col - j;
                        tempOrigtheta(i,j:end) = tempOrigtheta(i,suoyin1:suoyin2);
                        tempOrigabs(i,j:end) = tempOrigabs(i,suoyin1:suoyin2);
                        elseif fix(tempOrigtheta(i,j) / pi)  < 0 %&& col(i) - displace(row, col)<rowlong
                            suoyin1 = min(j - fixtem, col)
                            suoyin2 = suoyin1 - j + 1 ;

                        tempOrigtheta(i,1:j) = tempOrigtheta(i,suoyin2:suoyin1);
                        tempOrigabs(i,1:j) = tempOrigabs(i,suoyin2:suoyin1);
                    
                        end
                    end

                end
            elseif magup == 2
                for i = 2: row-1
                    for j = 2: col - 1
                        fixtem = fix(tempOrigtheta(i,j) / pi)
                        if fixtem > 0 %&& col(i) - displace(row, col)>0
                            
                            suoyin1 = max(i - fixtem, 1)
                            suoyin2 = suoyin1 + row - i;
                        tempOrigtheta(i:end,j) = tempOrigtheta(suoyin1:suoyin2,j);
                        tempOrigabs(i:end,j) = tempOrigabs(suoyin1:suoyin2,j);
                        elseif fix(tempOrigtheta(i,j) / pi)  < 0 %&& col(i) - displace(row, col)<rowlong
                            suoyin1 = min(i - fixtem, row)
                            suoyin2 = suoyin1 - i + 1 ;

                        tempOrigtheta(1:i,j) = tempOrigtheta(suoyin2:suoyin1,j);
                        tempOrigabs(1:i,j) = tempOrigabs(suoyin2:suoyin1,j);
                    
                        end
                    end
                end
            elseif magup == 1
                for i = 2: row-1
                    for j = 2: col - 1
                   % if row(i) - displace(row, col) < rowlong && row(i) - displace(row, col)>0
                   fixtem = fix(tempOrigtheta(i,j) / pi)
                       if fixtem > 0 %&& col(i) - displace(row, col)>0
                           
                                suoyin1 = max(i - fixtem, 1);
                                suoyin2 = max(j - fixtem, 1);
                                suoyin3 = min(i + fixtem, row);
                                suoyin4 = min(j + fixtem, col);
                                ichang = min(suoyin3 - i, i - suoyin1);
                                jchang = min(suoyin4 - j, j - suoyin2)
                            tempOrigtheta(suoyin3 - ichang:suoyin3,suoyin4- jchang:suoyin4 ) = tempOrigtheta(i-ichang:i,j-jchang:j);
                            tempOrigabs(suoyin3 - ichang:suoyin3,suoyin4- jchang:suoyin4) = tempOrigabs(i-ichang:i,j-jchang:j);
                            elseif fix(tempOrigtheta(i,j) / pi)  < 0 %&& col(i) - displace(row, col)<rowlong
                                suoyin1 = max(i + fixtem, 1);
                                suoyin2 = max(j + fixtem, 1);
                                suoyin3 = min(i - fixtem, row);
                                suoyin4 = min(j - fixtem, col);
                                ichang = min(suoyin3 - i, i - suoyin1);
                                jchang = min(suoyin4 - j, j - suoyin2)
                            tempOrigtheta(suoyin1:suoyin1 + ichang,suoyin2:suoyin2 + jchang) = tempOrigtheta(i:i+ ichang,j:j + jchang);
                            tempOrigabs(suoyin1:suoyin1 + ichang,suoyin2:suoyin2 + jchang) = tempOrigabs(i:i+ ichang,j:j + jchang);
                            
                       end
                    end
                end
            elseif magup == 3
                for i = 2: row-1
                    for j = 2: col - 1
                        fixtem = fix(tempOrigtheta(i,j) / pi)
                       if fixtem > 0 %&& col(i) - displace(row, col)>0
                           
                                suoyin1 = max(i - fixtem, 1);
                                suoyin2 = max(j - fixtem, 1);
                                suoyin3 = min(i + fixtem, row);
                                suoyin4 = min(j + fixtem, col);
                                ichang = min(suoyin3 - i, i - suoyin1);
                                jchang = min(suoyin4 - j, j - suoyin2)
                            tempOrigtheta(suoyin1:suoyin1 + ichang,suoyin4 - jchang:suoyin4) = tempOrigtheta(i:i + ichang,j - jchang:j);
                            tempOrigabs(suoyin1:suoyin1 + ichang,suoyin4 - jchang:suoyin4) = tempOrigabs(i:i + ichang,j - jchang:j);
                            elseif fix(tempOrigtheta(i,j) / pi)  < 0 %&& col(i) - displace(row, col)<rowlong
                                suoyin1 = max(i + fixtem, 1);
                                suoyin2 = max(j + fixtem, 1);
                                suoyin3 = min(i - fixtem, row);
                                suoyin4 = min(j - fixtem, col);
                                ichang = min(suoyin3 - i, i - suoyin1);
                                jchang = min(suoyin4 - j, j - suoyin2)
                            tempOrigtheta(suoyin3 - ichang:suoyin3,suoyin2:suoyin2 + jchang) = tempOrigtheta(i - ichang:i,j :j+ jchang);
                            tempOrigabs(suoyin3 - ichang:suoyin3,suoyin2:suoyin2 + jchang) = tempOrigabs(i - ichang:i,j :j+ jchang);
                       end                    
                    end
                end
            end
            %tempOrig = unwrap_phase(tempOrig);
            tempTransformOut = exp(1i*(phaseOfFrame+tempOrigtheta)).*tempOrigabs; %将放大后的相位加到傅里叶系数上

            curLevelFrame = reconLevel(tempTransformOut, level);%结果是复数矩阵，傅里叶变换相当于取傅里叶级数的复系数
            magnifiedLumaFFT(filtIDX{level,1}, filtIDX{level,2},frameIDX) = curLevelFrame + magnifiedLumaFFT(filtIDX{level,1}, filtIDX{level,2},frameIDX);
        end


    end
    %% Add unmolested lowpass residual
    level = numel(filters);
    for frameIDX = 1:nF 
        lowpassFrame = vidFFT(filtIDX{level,1},filtIDX{level,2},frameIDX).*croppedFilters{end}.^2;
        magnifiedLumaFFT(filtIDX{level,1},filtIDX{level,2},frameIDX) = magnifiedLumaFFT(filtIDX{level,1},filtIDX{level,2},frameIDX) + lowpassFrame;    
    end
    clear vidFFT;
  %  timeTic(5)=toc;
    fprintf('Begin Put frame in output\n');
    vid = vr.read([frames]);
    res = zeros(h,w,nC,nF,'uint8');
    for k = 1:nF
        magnifiedLuma = real(ifft2(ifftshift(magnifiedLumaFFT(:,:,k))));
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