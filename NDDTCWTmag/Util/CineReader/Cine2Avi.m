function Cine2Avi( inCine, outAvi, quality )
    if(nargin==2)
        quality = 90;
    end
    cr = CineReaderColor(inCine);
    vw = VideoWriter(outAvi);
    vw.Quality = quality;
    vw.FrameRate = 30;
    vw.open();
    for k = 1:cr.NumberOfFrames;
        if (mod(k,100) == 0)
            fprintf('Processing frame %d\n', k);
        end
        vw.writeVideo(im2uint8(cr.read(k)));
    end
    vw.close();

end

