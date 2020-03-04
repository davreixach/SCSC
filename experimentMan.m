%% ECCV 2020 Experiment Management
% David Reixach - IRI(CSIC-UPC) - 02.03.2020

%% Initialization

clear all, close all, clc
projectStartup('mcsc')
pythonStartup   % correct PY/MKL incompatibility

cd([project,'/SCSC'])


%% Run 1

% % F.benchmark 1
% exp = 1;
% filter_benchmark
% 
% clearvars -except project
% exp = 2;
% filter_benchmark
% 
% 
% % F.benchmark 2
% clearvars -except project
% exp = 1;
% filter_benchmark2
% 
% clearvars -except project
% exp = 2;
% filter_benchmark2
% 
% 
% % T.benchmark
% clearvars -except project
% exp = 1;
% time_benchmark
% 
% clearvars -except project
% exp = 2;
% time_benchmark
% 
% %% Run 2
% 
% % F.benchmark 1
% exp = 1;
% filter_benchmark
% 
% clearvars -except project
% exp = 2;
% filter_benchmark


% F.benchmark 2
% clearvars -except project
% exp = 1;
% filter_benchmark3
% 
% clearvars -except project
% exp = 2;
% filter_benchmark3

% clearvars -except project
exp = 3;
filter_benchmark3
