% Generate random tumor mask with textures
% Jianping Xu / 2021.12.17
clear
row = 96;
col = 96;
tumor_num = 12; % number of random tumors to generate
tum = zeros(row,col,1);
dirname = '.\Data\Natural_image\';     % Images of natural scene
dirs=dir([dirname,'*.jpg']);
num_BG = length(dirs);
filter_kernel = fspecial('gaussian',3,3); % filter for tumor
filter_kernel_2 = fspecial('gaussian',3,4);
for i = 1:tumor_num
    %% Get a random circle mask as tumor
    r = randi([3,8],1); % r: the radial of circule
    deta = 360.*rand(1,1);
    dx = randi([-20,20],1).*cosd(deta);
    dy = randi([-30,30],1).*sind(deta);
    nx = round(row/2 + dx);
    ny = round(row/2 + dy);
    mask_in = GgenCircle(row,r,[nx,ny]);
    r_ring = randi([2,4],1); % the radial of ring
    move_x = randi([-3,3],1);
    move_y = randi([-3,3],1);
    mask_out = GgenCircle(row,r+r_ring,[nx+move_x,ny+move_y]);
    move_x2 = randi([-3,3],1);
    move_y2 = randi([-3,3],1);
    mask_out_2 = GgenCircle(row,r+r_ring,[nx+move_x2,ny+move_y2]);
    mask_ring = mask_out-mask_in;
    mask_ring2 = mask_out_2-mask_out;
    mask_ring2(mask_ring2<0)=0;   

    %% Add texture in tumor
    ratio = 0.8; % the ratio of texture
    ratio_2 = randi([60,80],1)/100;
    ratio_3 = randi([80,90],1)/100;
    
    frame_i = randi([1,num_BG],1);
    filename = [dirname,dirs(frame_i).name];
    BG_img = double(rgb2gray(imread(filename)))/255;
    BG_img = abs(imresize(BG_img,[row,col],'nearest'));
    BG_img = imfilter(BG_img,filter_kernel','replicate');

    tumor_p1 = mask_in.*BG_img*ratio_2+(mask_in*(1-ratio_2));
    tumor_p2 = mask_ring.*BG_img*ratio+(mask_ring*(1-ratio));
    tumor_p3 = mask_ring2.*BG_img*ratio_3+(mask_ring2*(1-ratio_3));
    tumor = tumor_p1+tumor_p2+tumor_p3;
    tumor = imfilter(tumor,filter_kernel_2','replicate');
    tumor = tumor/(max(tumor(:)));
    %% Get healthy areas
    frame_j = randi([1,num_BG],1);
    filename2 = [dirname,dirs(frame_j).name];
    BG_img2 = double(rgb2gray(imread(filename2)))/255;
    BG_img2 = abs(imresize(BG_img2,[row,col],'nearest'));
    BG_img2 = imfilter(BG_img2,filter_kernel','replicate');
    tumor_whole = mask_in + mask_ring + mask_ring2;
    tumor_whole = imcomplement(tumor_whole);
    healthy_area = tumor_whole.*BG_img2*0.15+(tumor_whole*(1-0.15));
    healthy_area = imfilter(healthy_area,filter_kernel','replicate');
    healthy_area = (-1)* healthy_area/(max(max(healthy_area)));
    tumor_healthy = tumor + healthy_area;
    %%
    tum(:,:,i) = tumor_healthy;
end
tumor_mask = tum;
% save('./Data/simu/tumor_mask.mat','tumor_mask')
% imslide(tum)
montage(tum);
%% functions
function [ c ] = GgenCircle(w,r,center)
%	w: output size
%	r: the radial of circule 
%	center: the location of the center of the circle
[r1, c1] = meshgrid(1:w);
c = sqrt((r1-center(1)).^2 + (c1-center(2)).^2) <= r;
end
