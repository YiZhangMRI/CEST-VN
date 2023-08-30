% Modulate FastMRI dataset using simulated z-spectra.
% 2021.12.24
% created by by Jianping Xu

clear
addpath('.\functions')
col =96;
row =96;
dirname = '.\Data\Natural_image\'; % Images of natural scenes
for n = 1:2
    data_path = fullfile(['.\Data\FastMRI\',num2str(n),'\']);
    save_path = fullfile('.\Data\TrainData\',num2str(n));
    if ~exist(save_path)
        mkdir(save_path)
    end
    for loopj = 1:10
        load([data_path,'rawdata',num2str(loopj),'.mat']);
        load([data_path,'espirit',num2str(loopj),'.mat']);
        n_normal = fix(35*rand(1)+1); % load z-spectra with different simulation parameters randomly
        load(['.\Data\Z_spectra\z_normal_',num2str(n_normal),'.mat']) % z-spectrums for healthy tissue
        n_tumor = fix(31*rand(1)+1);
        load(['.\Data\Z_spectra\z_tumor_',num2str(n_tumor),'.mat']) % z-spectrums for tumor
        img = abs(reference);
        Input=img/(max(max(img)));
        tumor_det = tumor_detection(Input,col);
        tumor_det = double(imbinarize(tumor_det,0.9));
        S=0;
        nx=0;
        if max(tumor_det(:))~=0
            [nx,ny]=find(tumor_det~=0);
            nx = round(mean(nx));
            ny = round(mean(ny));
            S=sum(tumor_det(:));
            R = norm([col/2,col/2]-[nx,ny]);
            if R>32 % If too far away from center, it's not a tumor
                nx=0;
                S=0;
            end
        end
        tumor_mask = generate_tumor_mask(dirname,ny,nx,row,col,S,tumor_det);
 
%         figure;
%         subplot(1,3,1);  
%         imshow(Input);    
%         title('input','FontSize',20); 
%         subplot(1,3,2);  
%         imshow(tumor_det);   
%         title('tumor','FontSize',20);
%         subplot(1,3,3);  
%         imshow(tumor_mask);   
%         title('tumor mask','FontSize',20);
%         impixelinfo;
       %% 
        tumor = tumor_mask;
       %% Generate a z-spectrum mask according to structural image
        z_mask = ones(row,col,54);
        for k = 1:row
            for j = 1:col
               % for tumor
                if      tumor(k,j,:)>0.98
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_12(r,:),1,1,54); % relative higher APT
                elseif  0.97< tumor(k,j,:) && tumor(k,j,:)<=0.98
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_11(r,:),1,1,54);
                elseif  0.92< tumor(k,j,:) && tumor(k,j,:)<=0.97
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_10(r,:),1,1,54);
                elseif  0.86< tumor(k,j,:) && tumor(k,j,:)<=0.92
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_9(r,:),1,1,54);
                elseif  0.80< tumor(k,j,:) && tumor(k,j,:)<=0.86
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_8(r,:),1,1,54);
                elseif  0.74< tumor(k,j,:) && tumor(k,j,:)<=0.80
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_7(r,:),1,1,54);
                elseif  0.70< tumor(k,j,:) && tumor(k,j,:)<=0.74
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_6(r,:),1,1,54);
                elseif  0.65< tumor(k,j,:) && tumor(k,j,:)<=0.70
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_5(r,:),1,1,54);
                elseif  0.60< tumor(k,j,:) && tumor(k,j,:)<=0.65
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_4(r,:),1,1,54);
                elseif  0.30< tumor(k,j,:) && tumor(k,j,:)<=0.60
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_3(r,:),1,1,54);
                elseif  0.20< tumor(k,j,:) && tumor(k,j,:)<=0.30
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_2(r,:),1,1,54);
                elseif  0.0< tumor(k,j,:) && tumor(k,j,:)<=0.20
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_tumor.t_1(r,:),1,1,54);
                end
               % for healthy areas
                if      -0.3< tumor(k,j,:) && tumor(k,j,:)<=0.0
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_normal.n_9(r,:),1,1,54); % relative higher APT
                elseif  -0.5< tumor(k,j,:) && tumor(k,j,:)<=-0.3
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_normal.n_8(r,:),1,1,54);
                elseif  -0.70< tumor(k,j,:) && tumor(k,j,:)<=-0.5
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_normal.n_7(r,:),1,1,54);
                elseif  -0.90< tumor(k,j,:) && tumor(k,j,:)<=-0.70
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_normal.n_6(r,:),1,1,54);
                elseif  -0.94< tumor(k,j,:) && tumor(k,j,:)<=-0.90
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_normal.n_5(r,:),1,1,54);
                elseif  -0.96< tumor(k,j,:) && tumor(k,j,:)<=-0.94
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_normal.n_4(r,:),1,1,54);
                elseif  -0.97< tumor(k,j,:) && tumor(k,j,:)<=-0.96
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_normal.n_3(r,:),1,1,54);
                elseif  -0.98< tumor(k,j,:) && tumor(k,j,:)<=-0.97
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_normal.n_2(r,:),1,1,54);
                elseif  -1.00<= tumor(k,j,:) && tumor(k,j,:)<=-0.98
                        r = fix(9*rand(1)+1);
                        z_mask(k,j,:) = z_mask(k,j,:) .* reshape(z_normal.n_1(r,:),1,1,54); % relative lower APT
                end

            end
        end
        
        reference = reshape(reference,[row,col,1]) .* z_mask;
        img_temp = fftshift(ifft(ifftshift(rawdata,1),[],1),1); % [96 96 16]
        img_temp = fftshift(ifft(ifftshift(img_temp,2),[],2),2);
        img_temp = reshape(img_temp,[row,col,1,16]).* repmat(z_mask,[1,1,1,16]); % [96 96 54 16]
        img_temp = ifftshift(fft(fftshift(img_temp,1),[],1),1);
        rawdata = ifftshift(fft(fftshift(img_temp,2),[],2),2);

%         save(fullfile(save_path,['espirit',num2str(loopj),'.mat']),'reference','sensitivities')
%         save(fullfile(save_path,['rawdata',num2str(loopj),'.mat']),'rawdata')
    end
    disp(n)
end
rmpath('.\functions')