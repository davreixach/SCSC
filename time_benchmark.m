%% ECCV 2020 Datasets Management
% David Reixach - IRI(CSIC-UPC) - 09.02.2020
% Process Datasets

%% Initialization

% clear all, close all, clc
% projectStartup('mcsc')
% pythonStartup   % correct PY/MKL incompatibility
% 
% dbstop if error

rng('default')

%% Select data
% datasetsPath = '/home/david/Modular/Datasets/CVPR20/';
datasetsPath = '/home/dreixach/Modular/Datasets/CVPR20/';

% exp = 1;

nameCell = {'03_city_SCSC_';
            '03_fruit_SCSC_'};

dataCell = {[datasetsPath,'city.mat'];
        [datasetsPath,'fruit.mat'];
        [datasetsPath,'city_fruit_testing.mat']};

data = dataCell{exp};
name = nameCell{exp};
dataTest = dataCell{3};

load2(data,'S','b')
load2(dataTest,'S','btest')

%% set para

K = 100;

Ri = 10;
psf_s=[11,11]; 
psf_radius = floor( psf_s/2 );
precS = 1;
use_gpu = 1;
verbose = 'outer';

%% prepare data

b = squeeze(b);
btest = squeeze(btest);

padB = padarray(b, [psf_radius, 0], 0, 'both');
padBtest = padarray(btest, [psf_radius, 0], 0, 'both');

PARAtrain = auto_para_apg(Ri,K,psf_s,b,verbose,precS,use_gpu,1e-3);
PARAtest = auto_para_apg(Ri,K,psf_s,btest,verbose,precS,use_gpu,1e-3);

if (PARAtrain.precS ==1)
    b = single(b);
end
if (PARAtrain.gpu ==1)
    b = gpuArray(b);
end

%% CDL

for id_init = 1:5

    t1 = tic;
    [~,s_hat,R_D,R_Z] = apg_trainer2(padB,PARAtrain,b,padBtest,PARAtest,btest);
    td = toc(t1);    
    fprintf('\nDone training: %i! --> Time: %2.2f s\n\n', id_init, td)

    resTraining = R_D;
    resTesting =  R_Z;
    
    % save
    dataPath = [project,'/data/'];

    save2(sprintf([dataPath,name,'%02i','CDL_TrainResults.mat'],id_init),'dataset','resTraining','-noappend')
    save2(sprintf([dataPath,name,'%02i','CDL_TestResults.mat'],id_init),'dataset','resTesting','-noappend')
end

%% CSC

for id_init = 1:5
    
    resTesting = [];

    for it_z = linspace(1,5000,60)
        
        PARAtest.max_it_z = it_z;

        t2 = tic;
        [~,s_hat,R_Z] = apg_trainer(padBtest,PARAtest,btest,s_hat);    
        tc = toc(t2);    
        fprintf('\nDone testing K: %i! --> Time: %2.2f s, PSNR: %.2f, CR: %.2f\n\n\n', K, tc,R_Z.PSNR,R_Z.CR)

        resTesting =  [resTesting, R_Z];
        
        if R_Z.iter_code(end)<it_z
            break;
        end

    end

    % save
    dataPath = [project,'/data/'];

    save2(sprintf([dataPath,name,'%02i','CSC_TestResults.mat'],id_init),'dataset','resTesting','-noappend')
end
