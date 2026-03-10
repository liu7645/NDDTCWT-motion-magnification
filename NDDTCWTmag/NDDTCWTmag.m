clear all;
clc;
setPath

dataDir ='../data/'; 
filename = 'Bridge.avi';             
fprintf('Reproducing Bridge Results\n');

resultsDir10 = 'output/Bridgepyr';  
mkdir(resultsDir10); 
Bridge(dataDir, resultsDir10, filename);
runtime_method0 = toc; 

tic;
resultsDir11 = 'output/Bridgenddtcwt';  
mkdir(resultsDir11); 
Bridgeudtcwt(dataDir, resultsDir11, filename);
runtime_method2 = toc;