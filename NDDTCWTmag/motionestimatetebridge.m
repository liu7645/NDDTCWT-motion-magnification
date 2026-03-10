clear all;
clc;
setPath

dataDir ='../data/'; 

resultsDir = 'output/';  
mkdir(resultsDir);                                    
fprintf('Reproducing Bridge Results\n');
filename = 'Bridge.avi';

tic;
resultsDir1 = 'output/';  
mkdir(resultsDir1); 
Bridgedisplacement(dataDir, resultsDir1);
runtime_method1 = toc;


tic;
resultsDir3 = '../vi/output/bridge';  
mkdir(resultsDir3);   
Bridgedisplacementudt(dataDir, resultsDir3);
runtime_method3 = toc;


