function out = getFrameWithoutFocus( fig )
    % Similar to Matlab's in-built getframe, but without stealing focus
    % from another window. Useful when saving hundreds of frames for a
    % movie. 
    %
    % Neal Wadhwa, MIT 2016
    set(fig, 'Units', 'pixels', 'PaperPositionMode', 'auto');
    set(fig, 'InvertHardCopy', 'off');
     symbols = ['a':'z' 'A':'Z' '0':'9'];
     MAX_ST_LENGTH = 50;
     stLength = randi(MAX_ST_LENGTH);
     nums = randi(numel(symbols),[1 stLength]);
     st = symbols (nums);
     
     outFile = fullfile(tempdir, [st '.png']);
     print(fig, outFile, '-dpng');
     pause(0.01);
     out = imread(outFile);

end

