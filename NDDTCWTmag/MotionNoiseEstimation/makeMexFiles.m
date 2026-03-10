% Compiles computeMotionHalideMex.cpp into a mex file
% Precompiled binaries are available. They have been tested on  Windows 10, 
% OS X El Capitan and Ubuntu Linux 14.04 on MATLAB R2016a. These binaries 
% may work on other operating systems and versions of MATLAB.
%
% This mex file depends on Halide, a domain-specific language for
% image-processing. Source can be downloaded at
% https://github.com/halide/Halide and binaries at 
% https://github.com/halide/Halide/releases
HALIDE_DIR = '../../../../../halide';

addpath(genpath(HALIDE_DIR));
mex_halide('computeMotionHalideMex.cpp');