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

% exp = 1;

if exp == 2
    error('Stopping execution')
elseif exp == 3
    exp = 2;
end


nameCell = {'02_caltech_2_city_SCSC_';
            '02_caltech_2_fruit_SCSC_'};

dataCell = {[datasetsPath,'city.mat'];
        [datasetsPath,'fruit.mat'];
        [datasetsPath,'caltech_testing.mat']};

data = dataCell{exp};
name = nameCell{exp};
dataTest = dataCell{3};

load2(data,'S','b')
load2(dataTest,'S','btest_cell')

%% set para

% K_exp = [5,15,25,50,100,200];
K_exp = [100,200];


Ri = 10;
psf_s=[11,11]; 
psf_radius = floor( psf_s/2 );
precS = 1;
use_gpu = 0;
verbose = 'outer';
lambda_l1 = 1;

%% prepare data

b = squeeze(b);
padB = padarray(b, [psf_radius, 0], 0, 'both');

%% run experiment

resTraining = [];
resTesting = [];

for K = K_exp
    
    PARAtrain = auto_para_apg(Ri,K,psf_s,b,verbose,precS,use_gpu,1e-3,lambda_l1);

    if (PARAtrain.precS ==1)
        b = single(b);
    end
    if (PARAtrain.gpu ==1)
        b = gpuArray(b);
    end
    
    t1 = tic;
    [s,s_hat,R_D] = apg_trainer(padB,PARAtrain,b);
    td = toc(t1);    
    fprintf('\nDone training K: %i! --> Time: %2.2f s\n\n', K, td)
        
    R_D.K = K;
    resTraining = [resTraining, R_D];
    
    for j = 1:length(btest_cell)

        btest = squeeze(btest_cell{j});
        padBtest = padarray(btest, [psf_radius, 0], 0, 'both');        
        
        PARAtest = auto_para_apg(Ri,K,psf_s,btest,verbose,precS,use_gpu,1e-3,lambda_l1);
        
        s_1 = d2dsmall(s,PARAtrain);
        s_2 = dsmall2d(s_1,PARAtest);
        s_hat = fft2(s_2);

        t2 = tic;
        [~,s_hat,R_Z] = apg_trainer(padBtest,PARAtest,btest,s_hat);    
        tc = toc(t2);    
        fprintf('\nDone testing K: %i! --> Time: %2.2f s, PSNR: %.2f, CR: %.2f\n\n\n', K, tc,R_Z.PSNR,R_Z.CR)

        R_Z.K = K;
        resTesting = [resTesting, R_Z];
        
    end

end


%% save

dataPath = [project,'/data/'];

save2([dataPath,name,'TrainResults.mat'],'dataset','resTraining','-noappend')
save2([dataPath,name,'TestResults.mat'],'dataset','resTesting','-noappend')
