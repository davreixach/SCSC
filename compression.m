%% ECCV 2020 Compression
% David Reixach - IRI(CSIC-UPC) - 02.03.2020
% Choudhury Compression

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

nameCell = {'10_Choudhury_'};

dataCell = {[datasetsPath,'Choudhury_GT.mat']};

data = dataCell{exp};
name = nameCell{exp};
dataTest = dataCell{1};

load2(data,'S','b')
load2(dataTest,'S','btest')

%% set para

F_exp = [3,5,7,9,11];
K_exp = [5,15,25,50,100];
lmda_exp = [0.001,0.01,0.1,0.5,1,5,10];

Ri = 10;
precS = 1;
use_gpu = 1;
verbose = 'outer';

%% prepare data

b = squeeze(b);
btest = squeeze(btest);

%% run experiment

resTraining = [];
resTesting = [];

for F = F_exp

    psf_s=[F,F]; 
    psf_radius = floor( psf_s/2 );
    
    padB = padarray(b, [psf_radius, 0], 0, 'both');
    padBtest = padarray(btest, [psf_radius, 0], 0, 'both');
    
    for K = K_exp    
        for L = lmda_exp

            PARAtrain = auto_para_apg(Ri,K,psf_s,b,verbose,precS,use_gpu,1e-3,L);
            PARAtest = auto_para_apg(Ri,K,psf_s,btest,verbose,precS,use_gpu,1e-3,L);

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

            R_D.L = L;
            R_Z.L = L;

            R_D.K = K;
            R_Z.K = K;

            R_D.F = F;
            R_Z.F = F;

            resTraining = [resTraining, R_D];
            resTesting =  [resTesting, R_Z];

        end
    end
end

%% save

dataPath = [project,'/data/'];

save2([dataPath,name,'TrainResults.mat'],'dataset','resTraining','-noappend')
save2([dataPath,name,'TestResults.mat'],'dataset','resTesting','-noappend')
