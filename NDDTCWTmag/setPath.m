%restoredefaultpath

% Adds directories to MATLAB path

% Paths for the linear method
addpath(fullfile(pwd, 'Linear'));
addpath(genpath(fullfile(pwd, 'Util')));
addpath(fullfile(pwd, 'matlabPyrTools'));
addpath(fullfile(pwd, 'matlabPyrTools', 'MEX'));

% Paths for the phase-based method
addpath(fullfile(pwd, 'PhaseBased'));
addpath(fullfile(pwd, 'pyrToolsExt'));
addpath(fullfile(pwd, 'Filters'));
addpath(fullfile(pwd, 'data'));
addpath(fullfile(pwd, '../video/experiment/'));
% Paths for riesz-pyramid
addpath(genpath(fullfile(pwd, 'RieszPyramid')));
addpath(genpath(fullfile(pwd, 'unwrap phase')));
% Paths for ndtcwt
addpath(fullfile(pwd, 'NDDTCWT'));
% Paths for motion estimation
addpath(genpath(fullfile(pwd, 'MotionNoiseEstimation')));
addpath(genpath(fullfile(pwd, 'MotionMicroscopeScripts')));
addpath(fullfile(pwd, 'ICEEMDAN'));