%% ECCV 2020 Datasets Management
% David Reixach - IRI(CSIC-UPC) - 09.02.2020
% Process Datasets

%% Initialization

% clear all, close all, clc
% projectStartup('mcsc')
% pythonStartup   % correct PY/MKL incompatibility
% 
% dbstop if error
% 
% cd([project,'/SCSC'])

rng('default')

%% Select data
% datasetsPath = '/home/david/Modular/Datasets/CVPR20/';
datasetsPath = '/home/dreixach/Modular/Datasets/CVPR20/';

% exp = 3;

nameCell = {'02_city_2_SCSC_';
            '02_fruit_2_SCSC_'};

dataCell = {[datasetsPath,'city.mat'];
        [datasetsPath,'fruit.mat'];
        [datasetsPath,'city_fruit_testing.mat']};

data = dataCell{exp};
name = nameCell{exp};
dataTest = dataCell{3};

load2(data,'S','b')
load2(dataTest,'S','btest')

%% set para

K_exp = [5,15,25,50,100,200,400];

Ri = 10;
psf_s=[11,11]; 
psf_radius = floor( psf_s/2 );
precS = 1;
use_gpu = 1;
verbose = 'outer';
lambda_l1 = 1;

%% prepare data

b = squeeze(b);
btest = squeeze(btest);

padB = padarray(b, [psf_radius, 0], 0, 'both');
padBtest = padarray(btest, [psf_radius, 0], 0, 'both');

%% run experiment

resTraining = [];
resTesting = [];

for K = K_exp
    
    PARAtrain = auto_para_apg(Ri,K,psf_s,b,verbose,precS,use_gpu,1e-3,lambda_l1);
    PARAtest = auto_para_apg(Ri,K,psf_s,btest,verbose,precS,use_gpu,1e-3,lambda_l1);

    if (PARAtrain.precS ==1)
        b = single(b);
    end
    if (PARAtrain.gpu ==1)
        b = gpuArray(b);
    end
    
    t1 = tic;
    [~,s_hat,R_D] = apg_trainer(padB,PARAtrain,b);
    td = toc(t1);    
    fprintf('\nDone training K: %i! --> Time: %2.2f s\n\n', K, td)
    
    t2 = tic;
    [~,s_hat,R_Z] = apg_trainer(padBtest,PARAtest,btest,s_hat);    
    tc = toc(t2);    
    fprintf('\nDone testing K: %i! --> Time: %2.2f s, PSNR: %.2f, CR: %.2f\n\n\n', K, tc,R_Z.PSNR,R_Z.CR)
    
    R_D.K = K;
    R_Z.K = K;
    
    resTraining = [resTraining, R_D];
    resTesting =  [resTesting, R_Z];

end


%% save

dataPath = [project,'/data/'];

save2([dataPath,name,'TrainResults.mat'],'dataset','resTraining','-noappend')
save2([dataPath,name,'TestResults.mat'],'dataset','resTesting','-noappend')
