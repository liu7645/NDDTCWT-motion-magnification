function [ vid ] = Butterworth2ndOrder(vid, fl, fh)
    
    [B,A] = butter(2, 2*[fl, fh]);
   
    vid = filter(B,A, vid, [], 3);

end

