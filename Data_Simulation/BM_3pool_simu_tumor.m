% simulate z-spectra using Bloch-McConnell equation simulation
% 2021.12.17
% created by Xingwang Yong, modified by Jianping Xu

clear
% timeStamp = datestr(now,30);
% disp(timeStamp);

%% input
% options
isGaussWave = true; % RF pulse shape, gauss=1, rect=0
B1 = 2;
numStep = 64;       % divide a gauss RF-pulse into how many parts

% basic CEST parameter setup
gyr = 42.576;
T1m = 1;
R1m = 1/T1m;  % the longitudinal relaxation rate of semi-solid macromolecular proton pool, i.e., 1/T1m
CESTfreq = 431;
B0_multiply_gyr = CESTfreq/3.5;

scanned_offset = transpose(load('Offset frequencies for APT scans - 54pts.txt'));
scanned_offset(1) = [];  % get rid of infinite value
scanned_offset_sorted = sort(scanned_offset,'descend');
LocationCESTfreq_p = find(scanned_offset_sorted==CESTfreq);
LocationCESTfreq_n = find(scanned_offset_sorted==-CESTfreq);

paramStruct.R1m               = R1m;
paramStruct.B1                = B1;
paramStruct.gyr               = gyr;
paramStruct.isGaussWave       = isGaussWave;
% paramStruct.lineshapeLUTfname = lineshapeLUTfname;

%% simulate z-spectra using BM equation
% set the parameters used to simulate the z-spectrum
z_num = 10; % number of z-spectra in each grayscale
num_all = 0;
for j_MT=1:4
    M0_MT_1 = 0.06 + 0.01*(j_MT-1); % M0_MT = [0.06 ~ 0.10]*74.9 for tumor areas
    M0_MT_2 =  M0_MT_1 + 0.01;
    for j_T2=1:4
        T2_water_1 = 0.12 + 0.01*(j_T2-1); % T2_water = [0.12 ~ 0.18] for tumor areas
        T2_water_2 = T2_water_1 + 0.03;
        for j_T1=1:2
            T1_water_1 = 1.4 + 0.1*(j_T1-1); % T1_water = [1.4 ~ 1.7] for tumor areas
            T1_water_2 = T1_water_1 + 0.2;
            z_normal = [];
            for j_apt=1:12 % z-spectra for tumor tissue are divided into 12 grades
                disp(['Grade ',num2str(j_apt)])
                M_apt = 0.20 + 0.025*(j_apt-1); % M_apt = [0.20 ~ 0.475] for tumor areas
                zspectrum_54 = ones(z_num,54);
                for i=1:z_num
                    ub    =  [T1_water_2, T2_water_2, M0_MT_2, M_apt]; %%
                    lb    =  [T1_water_1, T2_water_1, M0_MT_1, M_apt];
                    m0_sim = lb + (ub - lb).*rand(size(lb));
            %         disp(m0_sim)
                    input = [];
                    input.paramStruct = paramStruct;
                    input.paramStruct.freq_selected = scanned_offset_sorted;

                    zspectrum_sim_all_freq = full_BM_simu_4pool(input, m0_sim, numStep);
                    zspectrum_54(i,2:54) = zspectrum_sim_all_freq;
                end
                eval(['z_tumor.t_',num2str(j_apt),'=','zspectrum_54',';']);
            end
%             save(['.\Data\simu\z_tumor_',num2str(num_all),'.mat'],'z_tumor');
            num_all = num_all+12*z_num;
            disp(['*** num of z-spectra : ',num2str(num_all)])
        end
    end
end

%% sub functions
function zspectrum_fit = full_BM_simu_4pool(input, m0, numStep)
% full Bloch-McConell equation simulation, 
% 4-pool: water, APT, NOE, and semisolid MT

%% input
% m0(1), T1w
% m0(2), T2w
% m0(3), T2m
% m0(4), M0m
% m0(5), k_mw

%% parse input
R1m               = input.paramStruct.R1m;
B1                = input.paramStruct.B1;
freq_selected     = input.paramStruct.freq_selected;  
gyr               = input.paramStruct.gyr;
isGaussWave       = input.paramStruct.isGaussWave;

%% pool specification 
% water pool (a)
M_0a = 1;     
R_2a = 1/m0(2);               
R_1a = 1/m0(1);

% APT, reference, DOI: 10.1002/nbm.4563
CESTfreq = 431;    % Hz, 431Hz==3.5ppm at 2.9T
M_0b = m0(4)/74.9;
R_2b = 1/0.002;
R_1b = 1/1;
k_ba = 30;
k_ab = k_ba * M_0b / M_0a;

% NOE, reference, DOI: 10.1002/nbm.4563
NOEfreq = -431;  % Hz, 431Hz==3.5ppm at 2.9T
M_0c = 0;
% M_0c = 0.5/74.9;
R_2c = 1/0.001;
R_1c = 1/1;
k_ca = 16;
k_ac = k_ca * M_0c / M_0a;

% MT, i.e. semisolid pool
R_2s = 1/10e-6;
M_0s = m0(3);
k_sa = 50;
k_as = k_sa * M_0s / M_0a;
R_1s = R1m; 

M0   = [0; 0; M_0a; 0; 0; M_0b; 0; 0; M_0c; 0; 0; M_0s; 1]; % a=water, b=APT, c=NOE, s=MT

%% pulse specification
[omega_1, crush_counter_init, crush_counter_max, step_size, numPulse,tp,td] = get_RF_pulse_mod(input, numStep);

%% simulation
off_resonance_range = freq_selected*2*pi;    % Hz to rad/s
M_z_w = zeros(size(off_resonance_range));
for iter = 1:numel(off_resonance_range)
    off_resonance = off_resonance_range(iter);
    delta_omega_a = off_resonance;
    delta_omega_b = off_resonance - CESTfreq*2*pi;
    delta_omega_c = off_resonance - NOEfreq*2*pi;
    delta_omega_s = off_resonance;
    
    M = M0;
    if isGaussWave
        crush_counter = crush_counter_init;
        for time_iter = 1: length(omega_1)
            A = [-R_2a-k_ab-k_ac-k_as,    -delta_omega_a   ,          0          ,     k_ba     ,          0         ,         0         ,     k_ca     ,          0         ,         0         ,     k_sa     ,          0         ,         0         ,     0    ;
                    delta_omega_a    , -R_2a-k_ab-k_ac-k_as,  omega_1(time_iter) ,       0      ,        k_ba        ,         0         ,       0      ,        k_ca        ,         0         ,       0      ,        k_sa        ,         0         ,     0    ;
                          0          , -omega_1(time_iter) , -R_1a-k_ab-k_ac-k_as,       0      ,          0         ,        k_ba       ,       0      ,          0         ,        k_ca       ,       0      ,          0         ,        k_sa       , R_1a*M_0a;
                         k_ab        ,          0          ,          0          ,  -R_2b-k_ba  ,   -delta_omega_b   ,         0         ,       0      ,          0         ,         0         ,       0      ,          0         ,         0         ,     0    ;
                          0          ,         k_ab        ,          0          , delta_omega_b,     -R_2b-k_ba     , omega_1(time_iter),       0      ,          0         ,         0         ,       0      ,          0         ,         0         ,     0    ;
                          0          ,          0          ,         k_ab        ,       0      , -omega_1(time_iter),     -R_1b-k_ba    ,       0      ,          0         ,         0         ,       0      ,          0         ,         0         , R_1b*M_0b;
                         k_ac        ,          0          ,          0          ,       0      ,          0         ,         0         ,  -R_2c-k_ca  ,   -delta_omega_c   ,         0         ,       0      ,          0         ,         0         ,     0    ;
                          0          ,         k_ac        ,          0          ,       0      ,          0         ,         0         , delta_omega_c,     -R_2c-k_ca     , omega_1(time_iter),       0      ,          0         ,         0         ,     0    ;
                          0          ,          0          ,         k_ac        ,       0      ,          0         ,         0         ,       0      , -omega_1(time_iter),     -R_1c-k_ca    ,       0      ,          0         ,         0         , R_1c*M_0c;
                         k_as        ,          0          ,          0          ,       0      ,          0         ,         0         ,       0      ,          0         ,         0         ,  -R_2s-k_sa  ,   -delta_omega_s   ,         0         ,     0    ;
                          0          ,         k_as        ,          0          ,       0      ,          0         ,         0         ,       0      ,          0         ,         0         , delta_omega_s,     -R_2s-k_sa     , omega_1(time_iter),     0    ;
                          0          ,          0          ,         k_as        ,       0      ,          0         ,         0         ,       0      ,          0         ,         0         ,       0      , -omega_1(time_iter),     -R_1s-k_sa    , R_1s*M_0s;
                          0          ,          0          ,          0          ,       0      ,          0         ,         0         ,       0      ,          0         ,         0         ,       0      ,          0         ,         0         ,     0    ;];

            if omega_1(time_iter) == 0
                crush_counter = crush_counter + 1;
            end
            if crush_counter == 1%RF_delay/step_size%omega_1(time_iter) == 0 % crushing
                M = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;]*M;
            elseif crush_counter == crush_counter_max
%                 crush_counter = 0;
                crush_counter = crush_counter_init;
            end
            M = expm(A*step_size)*(M);
        end
    else
        A = [-R_2a-k_ab-k_ac-k_as,    -delta_omega_a   ,          0          ,     k_ba     ,       0       ,      0     ,     k_ca     ,       0       ,      0     ,     k_sa     ,       0       ,      0     ,     0    ;
                delta_omega_a    , -R_2a-k_ab-k_ac-k_as,     gyr*B1*2*pi     ,       0      ,      k_ba     ,      0     ,       0      ,      k_ca     ,      0     ,       0      ,      k_sa     ,      0     ,     0    ;
                      0          ,     -gyr*B1*2*pi    , -R_1a-k_ab-k_ac-k_as,       0      ,       0       ,    k_ba    ,       0      ,       0       ,    k_ca    ,       0      ,       0       ,    k_sa    , R_1a*M_0a;
                     k_ab        ,          0          ,          0          ,  -R_2b-k_ba  , -delta_omega_b,      0     ,       0      ,       0       ,      0     ,       0      ,       0       ,      0     ,     0    ;
                      0          ,         k_ab        ,          0          , delta_omega_b,   -R_2b-k_ba  , gyr*B1*2*pi,       0      ,       0       ,      0     ,       0      ,       0       ,      0     ,     0    ;
                      0          ,          0          ,         k_ab        ,       0      ,  -gyr*B1*2*pi , -R_1b-k_ba ,       0      ,       0       ,      0     ,       0      ,       0       ,      0     , R_1b*M_0b;
                     k_ac        ,          0          ,          0          ,       0      ,       0       ,      0     ,  -R_2c-k_ca  , -delta_omega_c,      0     ,       0      ,       0       ,      0     ,     0    ;
                      0          ,         k_ac        ,          0          ,       0      ,       0       ,      0     , delta_omega_c,   -R_2c-k_ca  , gyr*B1*2*pi,       0      ,       0       ,      0     ,     0    ;
                      0          ,          0          ,         k_ac        ,       0      ,       0       ,      0     ,       0      ,  -gyr*B1*2*pi , -R_1c-k_ca ,       0      ,       0       ,      0     , R_1c*M_0c;
                     k_as        ,          0          ,          0          ,       0      ,       0       ,      0     ,       0      ,       0       ,      0     ,  -R_2s-k_sa  , -delta_omega_s,      0     ,     0    ;
                      0          ,         k_as        ,          0          ,       0      ,       0       ,      0     ,       0      ,       0       ,      0     , delta_omega_s,   -R_2s-k_sa  , gyr*B1*2*pi,     0    ;
                      0          ,          0          ,         k_as        ,       0      ,       0       ,      0     ,       0      ,       0       ,      0     ,       0      ,  -gyr*B1*2*pi , -R_1s-k_sa , R_1s*M_0s;
                      0          ,          0          ,          0          ,       0      ,       0       ,      0     ,       0      ,       0       ,      0     ,       0      ,       0       ,      0     ,     0    ;];

       
        % during crusher, no B1
        A1 = [-R_2a-k_ab-k_ac-k_as,    -delta_omega_a   ,          0          ,     k_ba     ,       0       ,     0     ,     k_ca     ,       0       ,     0     ,     k_sa     ,       0       ,     0     ,     0    ;
                 delta_omega_a    , -R_2a-k_ab-k_ac-k_as,          0          ,       0      ,      k_ba     ,     0     ,       0      ,      k_ca     ,     0     ,       0      ,      k_sa     ,     0     ,     0    ;
                       0          ,          0          , -R_1a-k_ab-k_ac-k_as,       0      ,       0       ,    k_ba   ,       0      ,       0       ,    k_ca   ,       0      ,       0       ,    k_sa   , R_1a*M_0a;
                      k_ab        ,          0          ,          0          ,  -R_2b-k_ba  , -delta_omega_b,     0     ,       0      ,       0       ,     0     ,       0      ,       0       ,     0     ,     0    ;
                       0          ,         k_ab        ,          0          , delta_omega_b,   -R_2b-k_ba  ,     0     ,       0      ,       0       ,     0     ,       0      ,       0       ,     0     ,     0    ;
                       0          ,          0          ,         k_ab        ,       0      ,       0       , -R_1b-k_ba,       0      ,       0       ,     0     ,       0      ,       0       ,     0     , R_1b*M_0b;
                      k_ac        ,          0          ,          0          ,       0      ,       0       ,     0     ,  -R_2c-k_ca  , -delta_omega_c,     0     ,       0      ,       0       ,     0     ,     0    ;
                       0          ,         k_ac        ,          0          ,       0      ,       0       ,     0     , delta_omega_c,   -R_2c-k_ca  ,     0     ,       0      ,       0       ,     0     ,     0    ;
                       0          ,          0          ,         k_ac        ,       0      ,       0       ,     0     ,       0      ,       0       , -R_1c-k_ca,       0      ,       0       ,     0     , R_1c*M_0c;
                      k_as        ,          0          ,          0          ,       0      ,       0       ,     0     ,       0      ,       0       ,     0     ,  -R_2s-k_sa  , -delta_omega_s,     0     ,     0    ;
                       0          ,         k_as        ,          0          ,       0      ,       0       ,     0     ,       0      ,       0       ,     0     , delta_omega_s,   -R_2s-k_sa  ,     0     ,     0    ;
                       0          ,          0          ,         k_as        ,       0      ,       0       ,     0     ,       0      ,       0       ,     0     ,       0      ,       0       , -R_1s-k_sa, R_1s*M_0s;
                       0          ,          0          ,          0          ,       0      ,       0       ,     0     ,       0      ,       0       ,     0     ,       0      ,       0       ,     0     ,     0    ;];

          
        for k = 1:numPulse
            M = expm(A*tp)*M;
            M = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;]*M;
            M = expm(A1*ceil(td/step_size)*step_size)*M; % relaxation
        end
    end
    M_z_w(iter) = M(3);
end

zspectrum_fit =  M_z_w;
end


function [omega_1, crush_counter_init, crush_counter_max, step_size, numPulse,tp,td] = get_RF_pulse_mod(input, numStep)

B1                = input.paramStruct.B1;
gyr               = input.paramStruct.gyr;
isGaussWave       = input.paramStruct.isGaussWave;


numPulse  = 10;       % number of pulse 
tp        = 0.097280; % 5120us*19, duration of each pulse, second
td        = 0.005;    % delay between each pulse, second
teile     = numStep;  % divide every pulse into how many pieces
tpulse    = tp/teile:tp/teile:tp;
step_size = tp/teile;

if ~isGaussWave % pulsed block
    omega_1_element = B1*2*pi*gyr*ones(1, tp/step_size);
else % pulsed Gaussian, pulse shape exported from IDEA's pulseTool
%     fname = 'GAUSS5120.B375.mat';
%     pulseStruct = load(fname);
%     original_RF_pulse = pulseStruct.RF_pulse;
%     original_RF_pulse = original_RF_pulse';
    

    original_RF_pulse = [0.000000000000000, 0.038578000000000, 0.039574001000000, 0.040591002000000, 0.041630000000000, 0.042691998000000, 0.043776002000000, 0.044883002000000, 0.046014000000000, 0.047168002000000, 0.048347000000000, 0.049550001000000, 0.050778002000000, 0.052030999000000, 0.053309001000000, 0.054614000000000, 0.055945002000000, 0.057303000000000, 0.058687001000000, 0.060100000000000, 0.061540000000000, 0.063008003000000, 0.064505003000000, 0.066030003000000, 0.067584999000000, 0.069169998000000, 0.070785001000000, 0.072430000000000, 0.074106000000000, 0.075814001000000, 0.077551998000000, 0.079323001000000... 
                        0.081125997000000, 0.082961999000000, 0.084830999000000, 0.086732998000000, 0.088669002000000, 0.090639003000000, 0.092643999000000, 0.094682999000000, 0.096758001000000, 0.098867998000000, 0.101014003000000, 0.103197001000000, 0.105416000000000, 0.107671998000000, 0.109964997000000, 0.112296000000000, 0.114665002000000, 0.117072001000000, 0.119516999000000, 0.122001998000000, 0.124525003000000, 0.127087995000000, 0.129691005000000, 0.132333994000000, 0.135016993000000, 0.137740999000000, 0.140505999000000, 0.143312007000000, 0.146158993000000, 0.149048001000000, 0.151978999000000, 0.154952005000000... 
                        0.157967001000000, 0.161025003000000, 0.164125994000000, 0.167269006000000, 0.170456007000000, 0.173685998000000, 0.176960006000000, 0.180277005000000, 0.183638006000000, 0.187042996000000, 0.190492004000000, 0.193985000000000, 0.197521999000000, 0.201104000000000, 0.204730004000000, 0.208400995000000, 0.212116003000000, 0.215875998000000, 0.219680995000000, 0.223529994000000, 0.227423996000000, 0.231362998000000, 0.235346004000000, 0.239373997000000, 0.243446007000000, 0.247563004000000, 0.251724988000000, 0.255930007000000, 0.260179996000000, 0.264474005000000, 0.268812001000000, 0.273193985000000... 
                        0.277619004000000, 0.282088012000000, 0.286599994000000, 0.291155010000000, 0.295753002000000, 0.300392985000000, 0.305076003000000, 0.309801012000000, 0.314567000000000, 0.319375008000000, 0.324223012000000, 0.329113007000000, 0.334042013000000, 0.339011997000000, 0.344020993000000, 0.349068999000000, 0.354155988000000, 0.359279990000000, 0.364443004000000, 0.369643003000000, 0.374879003000000, 0.380151004000000, 0.385459006000000, 0.390801996000000, 0.396180004000000, 0.401591003000000, 0.407034993000000, 0.412512004000000, 0.418020993000000, 0.423561007000000, 0.429131001000000, 0.434731007000000... 
                        0.440360010000000, 0.446016997000000, 0.451701999000000, 0.457412988000000, 0.463149995000000, 0.468912005000000, 0.474698991000000, 0.480508000000000, 0.486339986000000, 0.492193997000000, 0.498068005000000, 0.503961980000000, 0.509873986000000, 0.515805006000000, 0.521752000000000, 0.527714014000000, 0.533692002000000, 0.539682984000000, 0.545686007000000, 0.551701009000000, 0.557726979000000, 0.563762009000000, 0.569805026000000, 0.575855970000000, 0.581911981000000, 0.587972999000000, 0.594038010000000, 0.600106001000000, 0.606173992000000, 0.612242997000000, 0.618310988000000, 0.624375999000000... 
                        0.630437016000000, 0.636493981000000, 0.642543972000000, 0.648588002000000, 0.654622018000000, 0.660646021000000, 0.666658998000000, 0.672659993000000, 0.678646028000000, 0.684617996000000, 0.690572023000000, 0.696509004000000, 0.702426016000000, 0.708323002000000, 0.714197993000000, 0.720049024000000, 0.725875974000000, 0.731676996000000, 0.737450004000000, 0.743193984000000, 0.748907983000000, 0.754589975000000, 0.760239005000000, 0.765854001000000, 0.771434009000000, 0.776975989000000, 0.782478988000000, 0.787943006000000, 0.793365002000000, 0.798744977000000, 0.804080009000000, 0.809370995000000... 
                        0.814613998000000, 0.819809020000000, 0.824954987000000, 0.830051005000000, 0.835093021000000, 0.840083003000000, 0.845018029000000, 0.849896014000000, 0.854717016000000, 0.859480023000000, 0.864181995000000, 0.868824005000000, 0.873401999000000, 0.877916992000000, 0.882367015000000, 0.886750996000000, 0.891067028000000, 0.895314991000000, 0.899492979000000, 0.903599977000000, 0.907634020000000, 0.911596000000000, 0.915482998000000, 0.919294000000000, 0.923030019000000, 0.926687002000000, 0.930266023000000, 0.933764994000000, 0.937183022000000, 0.940519989000000, 0.943774998000000, 0.946945012000000... 
                        0.950030982000000, 0.953032017000000, 0.955946982000000, 0.958773971000000, 0.961513996000000, 0.964164972000000, 0.966726005000000, 0.969197989000000, 0.971578002000000, 0.973865986000000, 0.976063013000000, 0.978165984000000, 0.980175972000000, 0.982091010000000, 0.983911991000000, 0.985637009000000, 0.987267017000000, 0.988799989000000, 0.990235984000000, 0.991576016000000, 0.992816985000000, 0.993960977000000, 0.995006979000000, 0.995953023000000, 0.996801019000000, 0.997550011000000, 0.998199999000000, 0.998749018000000, 0.999198973000000, 0.999549985000000, 0.999800026000000, 0.999949992000000... 
                        1.000000000000000, 0.999949992000000, 0.999800026000000, 0.999549985000000, 0.999198973000000, 0.998749018000000, 0.998199999000000, 0.997550011000000, 0.996801019000000, 0.995953023000000, 0.995006979000000, 0.993960977000000, 0.992816985000000, 0.991576016000000, 0.990235984000000, 0.988799989000000, 0.987267017000000, 0.985637009000000, 0.983911991000000, 0.982091010000000, 0.980175972000000, 0.978165984000000, 0.976063013000000, 0.973865986000000, 0.971578002000000, 0.969197989000000, 0.966726005000000, 0.964164972000000, 0.961513996000000, 0.958773971000000, 0.955946982000000, 0.953032017000000... 
                        0.950030982000000, 0.946945012000000, 0.943774998000000, 0.940519989000000, 0.937183022000000, 0.933764994000000, 0.930266023000000, 0.926687002000000, 0.923030019000000, 0.919294000000000, 0.915482998000000, 0.911596000000000, 0.907634020000000, 0.903599977000000, 0.899492979000000, 0.895314991000000, 0.891067028000000, 0.886750996000000, 0.882367015000000, 0.877916992000000, 0.873401999000000, 0.868824005000000, 0.864181995000000, 0.859480023000000, 0.854717016000000, 0.849896014000000, 0.845018029000000, 0.840083003000000, 0.835093021000000, 0.830051005000000, 0.824954987000000, 0.819809020000000... 
                        0.814613998000000, 0.809370995000000, 0.804080009000000, 0.798744977000000, 0.793365002000000, 0.787943006000000, 0.782478988000000, 0.776975989000000, 0.771434009000000, 0.765854001000000, 0.760239005000000, 0.754589975000000, 0.748907983000000, 0.743193984000000, 0.737450004000000, 0.731676996000000, 0.725875974000000, 0.720049024000000, 0.714197993000000, 0.708323002000000, 0.702426016000000, 0.696509004000000, 0.690572023000000, 0.684617996000000, 0.678646028000000, 0.672659993000000, 0.666658998000000, 0.660646021000000, 0.654622018000000, 0.648588002000000, 0.642543972000000, 0.636493981000000... 
                        0.630437016000000, 0.624375999000000, 0.618310988000000, 0.612242997000000, 0.606173992000000, 0.600106001000000, 0.594038010000000, 0.587972999000000, 0.581911981000000, 0.575855970000000, 0.569805026000000, 0.563762009000000, 0.557726979000000, 0.551701009000000, 0.545686007000000, 0.539682984000000, 0.533692002000000, 0.527714014000000, 0.521752000000000, 0.515805006000000, 0.509873986000000, 0.503961980000000, 0.498068005000000, 0.492193997000000, 0.486339986000000, 0.480508000000000, 0.474698991000000, 0.468912005000000, 0.463149995000000, 0.457412988000000, 0.451701999000000, 0.446016997000000... 
                        0.440360010000000, 0.434731007000000, 0.429131001000000, 0.423561007000000, 0.418020993000000, 0.412512004000000, 0.407034993000000, 0.401591003000000, 0.396180004000000, 0.390801996000000, 0.385459006000000, 0.380151004000000, 0.374879003000000, 0.369643003000000, 0.364443004000000, 0.359279990000000, 0.354155988000000, 0.349068999000000, 0.344020993000000, 0.339011997000000, 0.334042013000000, 0.329113007000000, 0.324223012000000, 0.319375008000000, 0.314567000000000, 0.309801012000000, 0.305076003000000, 0.300392985000000, 0.295753002000000, 0.291155010000000, 0.286599994000000, 0.282088012000000... 
                        0.277619004000000, 0.273193985000000, 0.268812001000000, 0.264474005000000, 0.260179996000000, 0.255930007000000, 0.251724988000000, 0.247563004000000, 0.243446007000000, 0.239373997000000, 0.235346004000000, 0.231362998000000, 0.227423996000000, 0.223529994000000, 0.219680995000000, 0.215875998000000, 0.212116003000000, 0.208400995000000, 0.204730004000000, 0.201104000000000, 0.197521999000000, 0.193985000000000, 0.190492004000000, 0.187042996000000, 0.183638006000000, 0.180277005000000, 0.176960006000000, 0.173685998000000, 0.170456007000000, 0.167269006000000, 0.164125994000000, 0.161025003000000... 
                        0.157967001000000, 0.154952005000000, 0.151978999000000, 0.149048001000000, 0.146158993000000, 0.143312007000000, 0.140505999000000, 0.137740999000000, 0.135016993000000, 0.132333994000000, 0.129691005000000, 0.127087995000000, 0.124525003000000, 0.122001998000000, 0.119516999000000, 0.117072001000000, 0.114665002000000, 0.112296000000000, 0.109964997000000, 0.107671998000000, 0.105416000000000, 0.103197001000000, 0.101014003000000, 0.098867998000000, 0.096758001000000, 0.094682999000000, 0.092643999000000, 0.090639003000000, 0.088669002000000, 0.086732998000000, 0.084830999000000, 0.082961999000000... 
                        0.081125997000000, 0.079323001000000, 0.077551998000000, 0.075814001000000, 0.074106000000000, 0.072430000000000, 0.070785001000000, 0.069169998000000, 0.067584999000000, 0.066030003000000, 0.064505003000000, 0.063008003000000, 0.061540000000000, 0.060100000000000, 0.058687001000000, 0.057303000000000, 0.055945002000000, 0.054614000000000, 0.053309001000000, 0.052030999000000, 0.050778002000000, 0.049550001000000, 0.048347000000000, 0.047168002000000, 0.046014000000000, 0.044883002000000, 0.043776002000000, 0.042691998000000, 0.041630000000000, 0.040591002000000, 0.039574001000000, 0.000000000000000];

    if numel(original_RF_pulse)/numStep >= 1 % use less points
        RF_pulse = downsample(original_RF_pulse, numel(original_RF_pulse)/numStep);
    else
        RF_pulse = interp1(1:numel(original_RF_pulse), ...
                           original_RF_pulse, ...
                           linspace(1, numel(original_RF_pulse), numStep), ...
                           'cubic');
    end
     
    % flip-angle equivalent
%     omega_1_element = RF_pulse*(512/247.904572)*B1*2*pi*gyr;   
        
    % power equivalent
    rect_pulse_power = (B1*2*pi*gyr)^2*0.1; % amplitude normalized to 1, duration 0.1s
    gauss_pulse_power = trapz(tpulse, RF_pulse.^2);
    omega_1_element = RF_pulse*sqrt(rect_pulse_power/gauss_pulse_power);   
end

crush_counter_init = -(numel(omega_1_element) - nnz(omega_1_element));
crush_counter_max = ceil(td/step_size);
omega_1 = repmat([omega_1_element, zeros(1, crush_counter_max)],1,numPulse);

end