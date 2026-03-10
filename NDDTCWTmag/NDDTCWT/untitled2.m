 load('aaaaa.mat')
  load('bbbbb.mat')
   load('ay.mat')
    load('pyr.mat')

  J = 4;
  [Faf, Fsf] = AntonB;
  [af, sf] = dualfilt1;

z = NDixWav2DMEX(ay, 4, Fsf, sf,1);
