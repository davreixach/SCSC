function [s_curr,s_hat,resTrain,resTest,A_h,B_h] = apg_trainer2(Mtb,para,b,Mtbtest,paratest,btest,s_hat)      
%% Initialize variables
b_hat = fft2(Mtb);
clear Mtb

if nargin > 6   % dictionary is provided
    [W,~,z] = init_SWZ(para);
    solveDict = false;
else
    [W,s_small,z] = init_SWZ(para);
    s = dsmall2d(s_small,para);
    s_hat = fft2(s);
    solveDict = true;
end

%% Save all objective values and timings
A_h = [];
B_h=[];    
v=[];
y=[];
result = [];
resTest = [];
for s_i=1:para.N 
    if strcmp( para.verbose, 'all')||strcmp( para.verbose, 'outer')
        fprintf('--%d--\t',s_i) 
    end
    temp_b_hat = b_hat(:,:,s_i);
    temp_b = b(:,:,s_i); 
    t_Z = tic;%~~~~!!!
    [z_hat_si, W_si,z,iterZ] = updateZW_nmapg(z,W,temp_b_hat,s_hat,para,temp_b);
    W = (1-1/s_i)*W+1/s_i*W_si; 
    timeZ = toc(t_Z);

    if strcmp( para.verbose, 'all')||strcmp( para.verbose, 'outer')
        
        % NNz and CR
        NNZ = nnz(z);
        CR = prod(size(temp_b))/NNZ;
        
        [vd_hat]= s2d(W_si,s_hat,para); 
       [recX] = rec_fre(vd_hat,z_hat_si,para);
		ps = my_psnr(temp_b,recX,para);  
        fprintf('Y: no.iter: %d, psnr: %2.2f, nnz: %i, cr: %.2f\t', iterZ,ps,NNZ,CR)
        
        resTrain.PSNR_i(s_i) = ps;
        resTrain.NNZ_i(s_i) = NNZ;
        resTrain.CR_i(s_i) = CR;

    end
    
    resTrain.iter_code(s_i) = iterZ;
    resTrain.time_code(s_i) = timeZ;
    
    %% dic update
    if solveDict
        t_D =tic;%~~~~!!!
        y_hat_si = z2y(W_si,z_hat_si,para);
        if isempty(A_h)
            init_or_not = 1;
        else
            init_or_not = 2;
        end
        if para.gpu ==1
            [A_h,B_h] = hist_ocsc_gpu(temp_b_hat,y_hat_si, para, A_h,B_h,init_or_not);
        else
            [A_h,B_h] = hist_ocsc_cpu(temp_b_hat,y_hat_si, para, A_h,B_h,init_or_not);    
        end
        [s,s_hat,v,y,iterD] = updateD_ocsc_warm(para,A_h,B_h,v,y,s_hat);
        timeD =toc(t_D);


        if strcmp( para.verbose, 'all')||strcmp( para.verbose, 'outer')

            vd_hat = s2d(W_si,s_hat,para);
            [recX] = rec_fre(vd_hat,z_hat_si,para);
            ps = my_psnr(temp_b,recX,para);
           fprintf('S: no.iter: %d, psnr: %2.2f\n', iterD,ps)

        end
        
        resTrain.iter_dic(s_i) = iterD;
        resTrain.time_dic(s_i) = timeD;
        
    else
        fprintf('\n')    
    end  
    
    % run test
    [~,~,rTest] = apg_trainer(Mtbtest,paratest,btest,s_hat);
    
    resTest = [resTest,rTest];
    
end

if solveDict
    s_curr = d2dsmall(s,para);
else
    s_curr = nan;
end

resTrain.PSNR_i = gather(resTrain.PSNR_i);

resTrain.PSNR = mean(resTrain.PSNR_i);
resTrain.NNZ = sum(resTrain.NNZ_i);
resTrain.CR = mean(resTrain.CR_i);

end