%% ECCV 2020 Denoising
% David Reixach - IRI(CSIC-UPC) - 02.03.2020
% Choudhury Denoising

%% Initialization

clear all, close all, clc
projectStartup('mcsc')
pythonStartup   % correct PY/MKL incompatibility

dbstop if error

cd([project,'/SCSC'])

rng('default')

%% Select data
% datasetsPath = '/home/david/Modular/Datasets/CVPR20/';
datasetsPath = '/home/dreixach/Modular/Datasets/CVPR20/';

exp = 1;

nameCell = {'09_Choudhury_SCSC_'};

dataCell = {[datasetsPath,'fruit.mat'];
            [datasetsPath,'Choudhury_noised_2.mat'];
            [datasetsPath,'Choudhury_GT.mat']};

data = dataCell{exp};
name = nameCell{exp};
dataTest1 = dataCell{2};
dataTest2 = dataCell{3};

load2(data,'S','b')
load2(dataTest1,'S','btest1')
load2(dataTest2,'S','btest2')

%% set para

F = 11;
K = 100;
L = 1;

Ri = 10;
precS = 1;
use_gpu = 1;
verbose = 'outer';

%% prepare data

b = squeeze(b);
btest1 = squeeze(btest1);
btest2 = squeeze(btest2);

%% run experiment

resTraining = [];
resTesting = [];


psf_s=[F,F]; 
psf_radius = floor( psf_s/2 );

padB = padarray(b, [psf_radius, 0], 0, 'both');
padBtest = padarray(btest1, [psf_radius, 0], 0, 'both');

PARAtrain = auto_para_apg(Ri,K,psf_s,b,verbose,precS,use_gpu,1e-3,L);
PARAtest = auto_para_apg(Ri,K,psf_s,btest1,verbose,precS,use_gpu,1e-3,L);

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
[~,s_hat,R_Z] = apg_trainer3(padBtest,PARAtest,btest1,s_hat,btest2);    
tc = toc(t2);    
fprintf('\nDone denoising K: %i! --> Time: %2.2f s, PSNR: %.2f, CR: %.2f\n\n\n', K, tc,R_Z.PSNR_test,R_Z.CR)

R_D.L = L;
R_Z.L = L;

R_D.K = K;
R_Z.K = K;

R_D.F = F;
R_Z.F = F;

resTraining = [resTraining, R_D];
resTesting =  [resTesting, R_Z];


%% save

dataPath = [project,'/data/'];

save2([dataPath,name,'TrainResults.mat'],'dataset','resTraining','-noappend')
save2([dataPath,name,'TestResults.mat'],'dataset','resTesting','-noappend')
