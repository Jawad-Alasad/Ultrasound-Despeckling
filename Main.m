% JMI_sim_Rough_disc   : General simulation



close all
clear all


%%%%%%%%%%%%%%%%%%%%%%%%%%%  PARAMETERS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S1=8;   % block size vertocal
S2=8;   % block size horizontal
pow=4; % power in the kernel----- start from 1.5 up to 2.5 == inc=0.25
%vic=1;   % number of eigenvectors---- start from 5 down ;to 1
sz=1;   % image resizing, 256 or 128


% qmf = MakeONFilter('Daubechies',6);
% Level=2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     DATA     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load DiskRefPure.mat % Disc4096.mat
env=envP;
dn=img_Orig_disc(env, 16/sz, sz);

 NF=dn;
 
%%%load Disc4096N.mat

%  load DiskNoisy9.mat   %%% 9 is good for Arnoldi
% 
%  IM=img_Orig_disc(envN, 16/sz, sz);
 

 %IM=envN(1:32:max(size(envN)),:)/max(max(envN));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Noisy

rs0=[];
cnr0=[];
ssnr0=[];
snr0=[];
psnr0=[];
beta0=[];
fsim0=[];
mssim0=[];

%PCA
rspca=[];
cnrpca=[];
s_snrpca=[];
snrpca=[];
psnrpca=[];
betapca=[];
fsimpca=[];
mssimpca=[];
%QR
rsqr=[];
cnrqr=[];
s_snrqr=[];
snrqr=[];
psnrqr=[];
betaqr=[];
fsimqr=[];
mssimqr=[];

%SVD
rssvd=[];
cnrsvd=[];
s_snrsvd=[];
snrsvd=[];
psnrsvd=[];
betasvd=[];
fsimsvd=[];
mssimsvd=[];
% ARNOLDI
rsarn=[];
cnrarn=[];
s_snrarn=[];
snrarn=[];
psnrarn=[];
betaarn=[];
fsimarn=[];
mssimarn=[];
% LANCZOS
rslancz=[];
cnrlancz=[];
s_snrlancz=[];
snrlancz=[];
psnrlancz=[];
betalancz=[];
fsimlancz=[];
mssimlancz=[];
% ORHTONORMAL
rsorth=[];
cnrorth=[];
s_snrorth=[];
snrorth=[];
psnrorth=[];
betaorth=[];
fsimorth=[];
mssimorth=[];
% SCHUR
rsschur=[];
cnrschur=[];
s_snrschur=[];
snrschur=[];
psnrschur=[];
betaschur=[];
fsimschur=[];
mssimschur=[];
% DWT
rsdwt=[];
cnrdwt=[];
s_snrdwt=[];
snrdwt=[];
psnrdwt=[];
betadwt=[];
fsimdwt=[];
mssimdwt=[];

% DWT2D 
rsdwt2D=[];
cnrdwt2D=[];
s_snrdwt2D=[];
snrdwt2D=[];
psnrdwt2D=[];
betadwt2D=[];
fsimdwt2D=[];
mssimdwt2D=[];
% NLM
rsnlm=[];
cnrnlm=[];
s_snrnlm=[];
snrnlm=[];
psnrnlm=[];
betnlm=[];
fsimnlm=[];
mssimnlm=[];
% Wiener
rswien=[];
cnrwien=[];
s_snrwien=[];
snrwien=[];
psnrwien=[];
betawien=[];
fsimwien=[];
mssimwien=[];   
% PNLM
rspnlm=[];
cnrpnlm=[];
s_snrpnlm=[];
snrpnlm=[];
psnrpnlm=[];
betapnlm=[];
fsimpnlm=[];
mssimpnlm=[];
% ADF
rsadf=[];
cnradf=[];
s_snradf=[];
snradf=[];
psnradf=[];
betaadf=[];
fsimadf=[];
mssimadf=[];
% TVF
rstvf=[];
cnrtvf=[];
s_snrtvf=[];
snrtvf=[];
psnrtvf=[];
betatvf=[];
fsimtvf=[];
mssimtvf=[];
% SRAD
rssrad=[];
cnrsrad=[];
s_snrsrad=[];
snrsrad=[];
psnrsrad=[];
betasrad=[];
fsimsrad=[];
mssimsrad=[];
% FROST
rsfrost=[];
cnrfrost=[];
s_snrfrost=[];
snrfrost=[];
psnrfrost=[];
betafrost=[];
fsimfrost=[];
mssimfrost=[];
% KUAN
rskuan=[];
cnrkuan=[];
s_snrkuan=[];
snrkuan=[];
psnrkuan=[];
betakuan=[];
fsimkuan=[];
mssimkuan=[];
% LEE
rslee=[];
cnrlee=[];
s_snrlee=[];
snrlee=[];
psnrlee=[];
betalee=[];
fsimlee=[];
mssimlee=[];
% NCDF
rsncdf=[];
cnrncdf=[];
s_snrncdf=[];
snrncdf=[];
psnrncdf=[];
betancdf=[];
fsimncdf=[];
mssimncdf=[];
% BM3D
rsbm3d=[];
cnrbm3d=[];
s_snrbm3d=[];
snrbm3d=[];
psnrbm3d=[];
betabm3d=[];
fsimbm3d=[];
mssimbm3d=[];
% OBNLM
rsobnlm=[];
cnrobnlm=[];
s_snrobnlm=[];
snrobnlm=[];
psnrobnlm=[];
betaobnlm=[];
fsimobnlm=[];
mssimobnlm=[];
% DPAD
rsdpad=[];
cnrdpad=[];
s_snrdpad=[];
snrdpad=[];
psnrdpad=[];
betadpad=[];
fsimdpad=[];
mssimdpad=[];
% GAMMA
rsgamma=[];
cnrgamma=[];
s_snrgamma=[];
snrgamma=[];
psnrgamma=[];
betagamma=[];
fsimgamma=[];
mssimgamma=[];
% GNLDF
rsgnldf=[];
cnrgnldf=[];
s_snrgnldf=[];
snrgnldf=[];
psnrgnldf=[];
betagnldf=[];
fsimgnldf=[];
mssimgnldf=[];
% GMF
rsgmf=[];
cnrgmf=[];
s_snrgmf=[];
snrgmf=[];
psnrgmf=[];
betagmf=[];
fsimgmf=[];
mssimgmf=[];
% KONGRES
rskong=[];
cnrkong=[];
s_snrkong=[];
snrkong=[];
psnrkong=[];
betakong=[];
fsimkong=[];
mssimkong=[];
% SARSF
rssarsf=[];
cnrsarsf=[];
s_snrsarsf=[];
snrsarsf=[];
psnrsarsf=[];
betasarsf=[];
fsimsarsf=[];
mssimsarsf=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rrr=0;
for run=10%:100
rrr=rrr+1

IM=img_RoughSpeckle_disc(run,16/sz,sz);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  DESPECKLING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LG=log(IM+0.01);

%%%%%%%%%%%%%%%%%%%%%%%%%    Proposed Schemes    %%%%%%%%%%%%%%%%%%%%%%%%%%
 
% PCA

pcad=gsrbfilter(IM, 7);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QR

qrd= recOrthsvd(IM,S1,S2,pow);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVD

svdd=recObliqsvd(IM,S1,S2,pow);%recOrthTrans(IM,S1,S2,pow);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ARNOLDI

arnd=recOrtharnoldi(IM,S1,S2,pow);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LANCZOS

lanczd=recObliqarnoldi(IM,S1,S2,pow); %% reclancz(LG,S1,S2,pow); %%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ORTHONORMAL

orthd=LG;%recOrthqr(IM,S1,S2,pow);  %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCHUR

schurd=LG;%recObliqqr(IM,S1,S2,pow);  %%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DWT

dwtd=LG;%recdwt(LG,S1,S2,Level,qmf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DWT2D

dwt2Dd=LG;%ThreshWave2(LG,'S',0,0,Level,qmf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% NLM  definitely works in the Log domain

%mstd=std2(LG);
nlmd=LG;%NLmeansfilter(LG,5,3,mstd); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Wiener definitely works in the Log domain

%[wiend, noise] = wiener2(LG, [S1-1 S2-1]);
wiend=LG;%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PNLM
 mstd=std2(LG);
pnlmd=PNLM(LG,1,3,mstd,1);

%%%%%%%%%%%%%%%%%%%%%%%%%  Bench Mark Schemes    %%%%%%%%%%%%%%%%%%%%%%%%%%

% ADF0.005

adfd=IM;%anisodiff(IM, 30, 25,0.25 , 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TVF

tvfd=IM;%tvdenoise(IM,60,25);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SRAD

sradd=IM;%SRAD(IM,100,0.05, [0 0 436 182]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Frost

frostd=frost(IM);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kuan

kuand=IM;%kuan(IM, 5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee

leed=Lee_speckle1_removal(IM,[3 3], 3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NCDF

%[ncdfd, nIter, dTT] = NCDF(IM, 5);
ncdfd=IM;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BM3D

%[NA,bm3dd]= BM3D(1, IM, 60);
bm3dd=IM;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBNLM

obnlmd=IM;%OBNLM(IM,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DPAD

dpadd=IM;%DPAD( IM, 0.2, 25);   %%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GAMMA

gammad=IM;%Gamma_MAP( IM, 7, 'same', 3, 'symmetric', 0); %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GNLDF
gnldfd=gnldf(LG,30,0.25,'wregion');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GNLDF
gmfd=LG;%GMF(LG, [3 3], 3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GNLDF
kongd=IM;%kongres(IM, 7, 5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SARSF
sarsfd=IM;%sarsf(IM, [3 3], 5);

%sarsfd = perform_blsgsm_denoising(IM, options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



pcadec=(pcad); %% de-noised by pca
qrdec=(qrd); % de-noised by qr
svddec=(svdd); % de-noised by svd 
arndec=(arnd); % de-noised by arnoldi
lanczdec=(lanczd); % de-noised by qr
orthdec=exp(orthd); % de-noised by svd
schurdec=exp(schurd); % de-noised by arnoldi
dwtdec=exp(dwtd); % de-noised by dwt : Discrete Wavelet Transfrom
dwt2Ddec=exp(dwt2Dd);% de-noised by 2D dwt : Discrete Wavelet Transfrom
nlmdec=exp(nlmd); % NLM
wiendec=exp(wiend); % Wiener
pnlmdec=exp(pnlmd); % PNLM


adfdec=adfd;%exp(adfd); % ADF
tvfdec=tvfd;%exp(tvfd); % TVF
sraddec=sradd;%exp(sradd); % SRAD
frostdec=frostd;%exp(frostd); % Frost
kuandec=kuand;%exp(kuand); % Kuan
leedec=leed; % Lee
ncdfdec=ncdfd;  % NCDF
bm3ddec=bm3dd;  % BM3D
obnlmdec=obnlmd; % OBNLM
dpaddec=dpadd;
gammadec=gammad;

gnldfdec=exp(gnldfd);
gmfdec=exp(gmfd);
kongdec=kongd;
sarsfdec=sarsfd;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

estespca=abs(pcadec/max(pcadec(:)));  
estesqr=abs(qrdec/max(qrdec(:)));
estessvd=abs(svddec/max(svddec(:)));
estesarn=abs(arndec/max(arndec(:)));
esteslancz=abs(lanczdec/max(lanczdec(:)));
estesorth=abs(orthdec/max(orthdec(:)));
estesschur=abs(schurdec/max(schurdec(:)));
estesdwt=abs(dwtdec/max(dwtdec(:)));
estesdwt2D=abs(dwt2Ddec/max(dwt2Ddec(:)));
estesnlm=abs(nlmdec/max(nlmdec(:)));
esteswien=abs(wiendec/max(wiendec(:)));
estespnlm=abs(pnlmdec/max(pnlmdec(:)));


estesadf=abs(adfdec/max(adfdec(:)));
estestvf=abs(tvfdec/max(tvfdec(:)));
estessrad=abs(sraddec/max(sraddec(:)));
estesfrost=abs(frostdec/max(frostdec(:)));
esteskuan=abs(kuandec/max(kuandec(:)));
esteslee=abs(leedec/max(leedec(:)));
estesncdf=abs(ncdfdec/max(ncdfdec(:)));
estesbm3d=abs(bm3ddec/max(bm3ddec(:)));
estesobnlm=abs(obnlmdec/max(obnlmdec(:)));
estesdpad=abs(dpaddec/max(dpaddec(:)));
estesgamma=abs(gammadec/max(gammadec(:)));

estesgnldf=abs(gnldfdec/max(gnldfdec(:)));
estesgmf=abs(gmfdec/max(gmfdec(:)));
esteskong=abs(kongdec/max(kongdec(:)));
estessarsf=abs(sarsfdec/max(sarsfdec(:)));




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%S_SNR ; Complex form
SSNR0=S_SNR_func(IM); % S_SNR of original speckle noisy image

S_SNR_pca=S_SNR_func(pcadec);
S_SNR_qr=S_SNR_func(qrdec);
S_SNR_svd=S_SNR_func(svddec);
S_SNR_arn=S_SNR_func(arndec);
S_SNR_lancz=S_SNR_func(lanczdec);
S_SNR_orth=S_SNR_func(orthdec);
S_SNR_schur=S_SNR_func(schurdec);
S_SNR_dwt=S_SNR_func(dwtdec);
S_SNR_dwt2D=S_SNR_func(dwt2Ddec);
S_SNR_nlm=S_SNR_func(nlmdec);
S_SNR_wien=S_SNR_func(wiendec);
S_SNR_pnlm=S_SNR_func(pnlmdec);

S_SNR_adf=S_SNR_func(adfdec);
S_SNR_tvf=S_SNR_func(tvfdec);
S_SNR_srad=S_SNR_func(sraddec);
S_SNR_frost=S_SNR_func(frostdec);
S_SNR_kuan=S_SNR_func(kuandec);
S_SNR_lee=S_SNR_func(leedec);
S_SNR_ncdf=S_SNR_func(ncdfdec);
S_SNR_bm3d=S_SNR_func(bm3ddec);
S_SNR_obnlm=S_SNR_func(obnlmdec);
S_SNR_dpad=S_SNR_func(dpaddec);
S_SNR_gamma=S_SNR_func(gammadec);

S_SNR_gnldf=S_SNR_func(gnldfdec);
S_SNR_gmf=S_SNR_func(gmfdec);
S_SNR_kong=S_SNR_func(kongdec);
S_SNR_sarsf=S_SNR_func(sarsfdec);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SNR
SNR0=SNR_func(dn,IM);

SNR_pca=SNR_func(dn,estespca);
SNR_qr=SNR_func(dn,estesqr);
SNR_svd=SNR_func(dn,estessvd);
SNR_arn=SNR_func(dn,estesarn);
SNR_lancz=SNR_func(dn,esteslancz);
SNR_orth=SNR_func(dn,estesorth);
SNR_schur=SNR_func(dn,estesschur);
SNR_dwt=SNR_func(dn,estesdwt);
SNR_dwt2D=SNR_func(dn,estesdwt2D);
SNR_nlm=SNR_func(dn,estesnlm);
SNR_wien=SNR_func(dn,esteswien);
SNR_pnlm=SNR_func(dn,estespnlm);

SNR_adf=SNR_func(dn,estesadf);
SNR_tvf=SNR_func(dn,estestvf);
SNR_srad=SNR_func(dn,estessrad);
SNR_frost=SNR_func(dn,estesfrost);
SNR_kuan=SNR_func(dn,esteskuan);
SNR_lee=SNR_func(dn,esteslee);
SNR_ncdf=SNR_func(dn,estesncdf);
SNR_bm3d=SNR_func(dn,estesbm3d);
SNR_obnlm=SNR_func(dn,estesobnlm);
SNR_dpad=SNR_func(dn,estesdpad);
SNR_gamma=SNR_func(dn,estesgamma);

SNR_gnldf=SNR_func(dn,estesgnldf);
SNR_gmf=SNR_func(dn,estesgmf);
SNR_kong=SNR_func(dn,esteskong);
SNR_sarsf=SNR_func(dn,estessarsf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PSNR
PSNR0 = PSNR_func(dn,IM);

PSNR_pca = PSNR_func(dn,estespca);
PSNR_qr = PSNR_func(dn,estesqr);
PSNR_svd = PSNR_func(dn,estessvd);
PSNR_arn = PSNR_func(dn,estesarn);
PSNR_lancz = PSNR_func(dn,esteslancz);
PSNR_orth = PSNR_func(dn,estesorth);
PSNR_schur = PSNR_func(dn,estesschur);
PSNR_dwt = PSNR_func(dn,estesdwt);
PSNR_dwt2D = PSNR_func(dn,estesdwt2D);
PSNR_nlm = PSNR_func(dn,estesnlm);
PSNR_wien = PSNR_func(dn,esteswien);
PSNR_pnlm = PSNR_func(dn,estespnlm);

PSNR_adf = PSNR_func(dn,estesadf);
PSNR_tvf = PSNR_func(dn,estestvf);
PSNR_srad = PSNR_func(dn,estessrad);
PSNR_frost = PSNR_func(dn,estesfrost);
PSNR_kuan = PSNR_func(dn,esteskuan);
PSNR_lee = PSNR_func(dn,esteslee);
PSNR_ncdf = PSNR_func(dn,estesncdf);
PSNR_bm3d = PSNR_func(dn,estesbm3d);
PSNR_obnlm = PSNR_func(dn,estesobnlm);
PSNR_dpad = PSNR_func(dn,estesdpad);
PSNR_gamma = PSNR_func(dn,estesgamma);

PSNR_gnldf = PSNR_func(dn,estesgnldf);
PSNR_gmf = PSNR_func(dn,estesgmf);
PSNR_kong = PSNR_func(dn,esteskong);
PSNR_sarsf = PSNR_func(dn,estessarsf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BETA
BETA0 = BETA_func(dn,IM);

BETA_pca = BETA_func(dn,estespca);
BETA_qr = BETA_func(dn,estesqr);
BETA_svd = BETA_func(dn,estessvd);
BETA_arn = BETA_func(dn,estesarn);
BETA_lancz = BETA_func(dn,esteslancz);
BETA_orth = BETA_func(dn,estesorth);
BETA_schur = BETA_func(dn,estesschur);
BETA_dwt = BETA_func(dn,estesdwt);
BETA_dwt2D = BETA_func(dn,estesdwt2D);
BETA_nlm = BETA_func(dn,estesnlm);
BETA_wien = BETA_func(dn,esteswien);
BETA_pnlm = BETA_func(dn,estespnlm);

BETA_adf = BETA_func(dn,estesadf);
BETA_tvf = BETA_func(dn,estestvf);
BETA_srad = BETA_func(dn,estessrad);
BETA_frost = BETA_func(dn,estesfrost);
BETA_kuan = BETA_func(dn,esteskuan);
BETA_lee = BETA_func(dn,esteslee);
BETA_ncdf = BETA_func(dn,estesncdf);
BETA_bm3d = BETA_func(dn,estesbm3d);
BETA_obnlm = BETA_func(dn,estesobnlm);
BETA_dpad = BETA_func(dn,estesdpad);
BETA_gamma = BETA_func(dn,estesgamma);

BETA_gnldf = BETA_func(dn,estesgnldf);
BETA_gmf = BETA_func(dn,estesgmf);
BETA_kong = BETA_func(dn,esteskong);
BETA_sarsf = BETA_func(dn,estessarsf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   RESOLUTION

RES0 = RES_func(IM);   % Resolution of original speckle noisy image

RES_pca = RES_func(estespca);
RES_qr = RES_func(estesqr);
RES_svd = RES_func(estessvd);
RES_arn = RES_func(estesarn);
RES_lancz = RES_func(esteslancz);
RES_orth = RES_func(estesorth);
RES_schur = RES_func(estesschur);
RES_dwt = RES_func(estesdwt);
RES_dwt2D = RES_func(estesdwt2D);
RES_nlm = RES_func(estesnlm);
RES_wien = RES_func(esteswien);
RES_pnlm = RES_func(estespnlm);

RES_adf = RES_func(estesadf);
RES_tvf = RES_func(estestvf);
RES_srad = RES_func(estessrad);
RES_frost = RES_func(estesfrost);
RES_kuan = RES_func(esteskuan);
RES_lee = RES_func(esteslee);
RES_ncdf = RES_func(estesncdf);
RES_bm3d = RES_func(estesbm3d);
RES_obnlm = RES_func(estesobnlm);
RES_dpad = RES_func(estesdpad);
RES_gamma = RES_func(estesgamma);

RES_gnldf = RES_func(estesgnldf);
RES_gmf = RES_func(estesgmf);
RES_kong = RES_func(esteskong);
RES_sarsf = RES_func(estessarsf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   CNR
CNR0 = CNR_func(IM);

CNR_pca = CNR_func(estespca);
CNR_qr = CNR_func(estesqr);
CNR_svd = CNR_func(estessvd);
CNR_arn = CNR_func(estesarn);
CNR_lancz = CNR_func(esteslancz);
CNR_orth = CNR_func(estesorth);
CNR_schur = CNR_func(estesschur);
CNR_dwt = CNR_func(estesdwt);
CNR_dwt2D = CNR_func(estesdwt2D);
CNR_nlm = CNR_func(estesnlm);
CNR_wien = CNR_func(esteswien);
CNR_pnlm = CNR_func(estespnlm);

CNR_adf = CNR_func(estesadf);
CNR_tvf = CNR_func(estestvf);
CNR_srad = CNR_func(estessrad);
CNR_frost = CNR_func(estesfrost);
CNR_kuan = CNR_func(esteskuan);
CNR_lee = CNR_func(esteslee);
CNR_ncdf = CNR_func(estesncdf);
CNR_bm3d = CNR_func(estesbm3d);
CNR_obnlm = CNR_func(estesobnlm);
CNR_dpad = CNR_func(estesdpad);
CNR_gamma = CNR_func(estesgamma);

CNR_gnldf = CNR_func(estesgnldf);
CNR_gmf = CNR_func(estesgmf);
CNR_kong = CNR_func(esteskong);
CNR_sarsf = CNR_func(estessarsf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   FSIM
FSIM0 = FeatureSIM(dn,IM);

FSIM_pca = FeatureSIM(dn,estespca);
FSIM_qr = FeatureSIM(dn,estesqr);
FSIM_svd = FeatureSIM(dn,estessvd);
FSIM_arn = FeatureSIM(dn,estesarn);
FSIM_lancz = FeatureSIM(dn,esteslancz);
FSIM_orth = FeatureSIM(dn,estesorth);
FSIM_schur = FeatureSIM(dn,estesschur);
FSIM_dwt = FeatureSIM(dn,estesdwt);
FSIM_dwt2D = FeatureSIM(dn,estesdwt2D);
FSIM_nlm = FeatureSIM(dn,estesnlm);
FSIM_wien = FeatureSIM(dn,esteswien);
FSIM_pnlm = FeatureSIM(dn,estespnlm);

FSIM_adf = FeatureSIM(dn,estesadf);
FSIM_tvf = FeatureSIM(dn,estestvf);
FSIM_srad = FeatureSIM(dn,estessrad);
FSIM_frost = FeatureSIM(dn,estesfrost);
FSIM_kuan = FeatureSIM(dn,esteskuan);
FSIM_lee = FeatureSIM(dn,esteslee);
FSIM_ncdf = FeatureSIM(dn,estesncdf);
FSIM_bm3d = FeatureSIM(dn,estesbm3d);
FSIM_obnlm = FeatureSIM(dn,estesobnlm);
FSIM_dpad = FeatureSIM(dn,estesdpad);
FSIM_gamma = FeatureSIM(dn,estesgamma);

FSIM_gnldf = FeatureSIM(dn,estesgnldf);
FSIM_gmf = FeatureSIM(dn,estesgmf);
FSIM_kong = FeatureSIM(dn,esteskong);
FSIM_sarsf = FeatureSIM(dn,estessarsf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   MSSIM

MSSIM0 = MSSIM(dn,IM);

MSSIM_pca = MSSIM(dn,estespca);
MSSIM_qr = MSSIM(dn,estesqr);
MSSIM_svd = MSSIM(dn,estessvd);
MSSIM_arn = MSSIM(dn,estesarn);
MSSIM_lancz = MSSIM(dn,esteslancz);
MSSIM_orth = MSSIM(dn,estesorth);
MSSIM_schur = MSSIM(dn,estesschur);
MSSIM_dwt = MSSIM(dn,estesdwt);
MSSIM_dwt2D = MSSIM(dn,estesdwt2D);
MSSIM_nlm = MSSIM(dn,estesnlm);
MSSIM_wien = MSSIM(dn,esteswien);
MSSIM_pnlm = MSSIM(dn,estespnlm);

MSSIM_adf = MSSIM(dn,estesadf);
MSSIM_tvf = MSSIM(dn,estestvf);
MSSIM_srad = MSSIM(dn,estessrad);
MSSIM_frost = MSSIM(dn,estesfrost);
MSSIM_kuan = MSSIM(dn,esteskuan);
MSSIM_lee = MSSIM(dn,esteslee);
MSSIM_ncdf = MSSIM(dn,estesncdf);
MSSIM_bm3d = MSSIM(dn,estesbm3d);
MSSIM_obnlm = MSSIM(dn,estesobnlm);
MSSIM_dpad = MSSIM(dn,estesdpad);
MSSIM_gamma = MSSIM(dn,estesgamma);

MSSIM_gnldf = MSSIM(dn,estesgnldf);
MSSIM_gmf = MSSIM(dn,estesgmf);
MSSIM_kong = MSSIM(dn,esteskong);
MSSIM_sarsf = MSSIM(dn,estessarsf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original Noisy

rs0=[rs0 RES0];
cnr0=[cnr0 CNR0];
ssnr0=[ssnr0 SSNR0];
snr0=[snr0 SNR0];
psnr0=[psnr0 PSNR0];
beta0=[beta0 BETA0];
fsim0=[fsim0 FSIM0];
mssim0=[mssim0 MSSIM0];
% PCA
rspca=[rspca RES_pca];
cnrpca=[cnrpca CNR_pca];
s_snrpca=[s_snrpca S_SNR_pca];
snrpca=[snrpca SNR_pca];
psnrpca=[psnrpca PSNR_pca];
betapca=[betapca BETA_pca];
fsimpca=[fsimpca FSIM_pca];
mssimpca=[mssimpca MSSIM_pca];
% QR
rsqr=[rsqr RES_qr];
cnrqr=[cnrqr CNR_qr];
s_snrqr=[s_snrqr S_SNR_qr];
snrqr=[snrqr SNR_qr];
psnrqr=[psnrqr PSNR_qr];
betaqr=[betaqr BETA_qr];
fsimqr=[fsimqr FSIM_qr];
mssimqr=[mssimqr MSSIM_qr];

% SVD
rssvd=[rssvd RES_svd];
cnrsvd=[cnrsvd CNR_svd];
s_snrsvd=[s_snrsvd S_SNR_svd];
snrsvd=[snrsvd SNR_svd];
psnrsvd=[psnrsvd PSNR_svd];
betasvd=[betasvd BETA_svd];
fsimsvd=[fsimsvd FSIM_svd];
mssimsvd=[mssimsvd MSSIM_svd];

% ARNOLDI
rsarn=[rsarn RES_arn];
cnrarn=[cnrarn CNR_arn];
s_snrarn=[s_snrarn S_SNR_arn];
snrarn=[snrarn SNR_arn];
psnrarn=[psnrarn PSNR_arn];
betaarn=[betaarn BETA_arn];
fsimarn=[fsimarn FSIM_arn];
mssimarn=[mssimarn MSSIM_arn];

% LANCZOS
rslancz=[rslancz RES_lancz];
cnrlancz=[cnrlancz CNR_lancz];
s_snrlancz=[s_snrlancz S_SNR_lancz];
snrlancz=[snrlancz SNR_lancz];
psnrlancz=[psnrlancz PSNR_lancz];
betalancz=[betalancz BETA_lancz];
fsimlancz=[fsimlancz FSIM_lancz];
mssimlancz=[mssimlancz MSSIM_lancz];

% ORTH
rsorth=[rsorth RES_orth];
cnrorth=[cnrorth CNR_orth];
s_snrorth=[s_snrorth S_SNR_orth];
snrorth=[snrorth SNR_orth];
psnrorth=[psnrorth PSNR_orth];
betaorth=[betaorth BETA_orth];
fsimorth=[fsimorth FSIM_orth];
mssimorth=[mssimorth MSSIM_orth];

% SCHUR
rsschur=[rsschur RES_schur];
cnrschur=[cnrschur CNR_schur];
s_snrschur=[s_snrschur S_SNR_schur];
snrschur=[snrschur SNR_schur];
psnrschur=[psnrschur PSNR_schur];
betaschur=[betaschur BETA_schur];
fsimschur=[fsimschur FSIM_schur];
mssimschur=[mssimschur MSSIM_schur];

% DWT
rsdwt=[rsdwt RES_dwt];
cnrdwt=[cnrdwt CNR_dwt];
s_snrdwt=[s_snrdwt S_SNR_dwt];
snrdwt=[snrdwt SNR_dwt];
psnrdwt=[psnrdwt PSNR_dwt];
betadwt=[betadwt BETA_dwt];
fsimdwt=[fsimdwt FSIM_dwt];
mssimdwt=[mssimdwt MSSIM_dwt];
% DWT2D
rsdwt2D=[rsdwt2D RES_dwt2D];
cnrdwt2D=[cnrdwt2D CNR_dwt2D];
s_snrdwt2D=[s_snrdwt2D S_SNR_dwt2D];
snrdwt2D=[snrdwt2D SNR_dwt2D];
psnrdwt2D=[psnrdwt2D PSNR_dwt2D];
betadwt2D=[betadwt2D BETA_dwt2D];
fsimdwt2D=[fsimdwt2D FSIM_dwt2D];
mssimdwt2D=[mssimdwt2D MSSIM_dwt2D];
% NLM
rsnlm=[rsnlm RES_nlm];
cnrnlm=[cnrnlm CNR_nlm];
s_snrnlm=[s_snrnlm S_SNR_nlm];
snrnlm=[snrnlm SNR_nlm];
psnrnlm=[psnrnlm PSNR_nlm];
betnlm=[betnlm BETA_nlm];
fsimnlm=[fsimnlm FSIM_nlm];
mssimnlm=[mssimnlm MSSIM_nlm];

% Wiener
rswien=[rswien RES_wien];
cnrwien=[cnrwien CNR_wien];
s_snrwien=[s_snrwien S_SNR_wien];
snrwien=[snrwien SNR_wien];
psnrwien=[psnrwien PSNR_wien];
betawien=[betawien BETA_wien];
fsimwien=[fsimwien FSIM_wien];
mssimwien=[mssimwien MSSIM_wien];

% PNLM
rspnlm=[rspnlm RES_pnlm];
cnrpnlm=[cnrpnlm CNR_pnlm];
s_snrpnlm=[s_snrpnlm S_SNR_pnlm];
snrpnlm=[snrpnlm SNR_pnlm];
psnrpnlm=[psnrpnlm PSNR_pnlm];
betapnlm=[betapnlm BETA_pnlm];
fsimpnlm=[fsimpnlm FSIM_pnlm];
mssimpnlm=[mssimpnlm MSSIM_pnlm];

% ADF
rsadf=[rsadf RES_adf];
cnradf=[cnradf CNR_adf];
s_snradf=[s_snradf S_SNR_adf];
snradf=[snradf SNR_adf];
psnradf=[psnradf PSNR_adf];
betaadf=[betaadf BETA_adf];
fsimadf=[fsimadf FSIM_adf];
mssimadf=[mssimadf MSSIM_adf];

% TVF
rstvf=[rstvf RES_tvf];
cnrtvf=[cnrtvf CNR_tvf];
s_snrtvf=[s_snrtvf S_SNR_tvf];
snrtvf=[snrtvf SNR_tvf];
psnrtvf=[psnrtvf PSNR_tvf];
betatvf=[betatvf BETA_tvf];
fsimtvf=[fsimtvf FSIM_tvf];
mssimtvf=[mssimtvf MSSIM_tvf];

% SRAD
rssrad=[rssrad RES_srad];
cnrsrad=[cnrsrad CNR_srad];
s_snrsrad=[s_snrsrad S_SNR_srad];
snrsrad=[snrsrad SNR_srad];
psnrsrad=[psnrsrad PSNR_srad];
betasrad=[betasrad BETA_srad];
fsimsrad=[fsimsrad FSIM_srad];
mssimsrad=[mssimsrad MSSIM_srad];

% FROST
rsfrost=[rsfrost RES_frost];
cnrfrost=[cnrfrost CNR_frost];
s_snrfrost=[s_snrfrost S_SNR_frost];
snrfrost=[snrfrost SNR_frost];
psnrfrost=[psnrfrost PSNR_frost];
betafrost=[betafrost BETA_frost];
fsimfrost=[fsimfrost FSIM_frost];
mssimfrost=[mssimfrost MSSIM_frost];

% KUAN
rskuan=[rskuan RES_kuan];
cnrkuan=[cnrkuan CNR_kuan];
s_snrkuan=[s_snrkuan S_SNR_kuan];
snrkuan=[snrkuan SNR_kuan];
psnrkuan=[psnrkuan PSNR_kuan];
betakuan=[betakuan BETA_kuan];
fsimkuan=[fsimkuan FSIM_kuan];
mssimkuan=[mssimkuan MSSIM_kuan];
% LEE
rslee=[rslee RES_lee];
cnrlee=[cnrlee CNR_lee];
s_snrlee=[s_snrlee S_SNR_lee];
snrlee=[snrlee SNR_lee];
psnrlee=[psnrlee PSNR_lee];
betalee=[betalee BETA_lee];
fsimlee=[fsimlee FSIM_lee];
mssimlee=[mssimlee MSSIM_lee];

% NCDF
rsncdf=[rsncdf RES_ncdf];
cnrncdf=[cnrncdf CNR_ncdf];
s_snrncdf=[s_snrncdf S_SNR_ncdf];
snrncdf=[snrncdf SNR_ncdf];
psnrncdf=[psnrncdf PSNR_ncdf];
betancdf=[betancdf BETA_ncdf];
fsimncdf=[fsimncdf FSIM_ncdf];
mssimncdf=[mssimncdf MSSIM_ncdf];

% NCDF
rsbm3d=[rsbm3d RES_bm3d];
cnrbm3d=[cnrbm3d CNR_bm3d];
s_snrbm3d=[s_snrbm3d S_SNR_bm3d];
snrbm3d=[snrbm3d SNR_bm3d];
psnrbm3d=[psnrbm3d PSNR_bm3d];
betabm3d=[betabm3d BETA_bm3d];
fsimbm3d=[fsimbm3d FSIM_bm3d];
mssimbm3d=[mssimbm3d MSSIM_bm3d];

% OBNLM
rsobnlm=[rsobnlm RES_obnlm];
cnrobnlm=[cnrobnlm CNR_obnlm];
s_snrobnlm=[s_snrobnlm S_SNR_obnlm];
snrobnlm=[snrobnlm SNR_obnlm];
psnrobnlm=[psnrobnlm PSNR_obnlm];
betaobnlm=[betaobnlm BETA_obnlm];
fsimobnlm=[fsimobnlm FSIM_obnlm];
mssimobnlm=[mssimobnlm MSSIM_obnlm];

% DPAD
rsdpad=[rsdpad RES_dpad];
cnrdpad=[cnrdpad CNR_dpad];
s_snrdpad=[s_snrdpad S_SNR_dpad];
snrdpad=[snrdpad SNR_dpad];
psnrdpad=[psnrdpad PSNR_dpad];
betadpad=[betadpad BETA_dpad];
fsimdpad=[fsimdpad FSIM_dpad];
mssimdpad=[mssimdpad MSSIM_dpad];

% GAMMA
rsgamma=[rsgamma RES_gamma];
cnrgamma=[cnrgamma CNR_gamma];
s_snrgamma=[s_snrgamma S_SNR_gamma];
snrgamma=[snrgamma SNR_gamma];
psnrgamma=[psnrgamma PSNR_gamma];
betagamma=[betagamma BETA_gamma];
fsimgamma=[fsimgamma FSIM_gamma];
mssimgamma=[mssimgamma MSSIM_gamma];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% GNLDF
rsgnldf=[rsgnldf RES_gnldf];
cnrgnldf=[cnrgnldf CNR_gnldf];
s_snrgnldf=[s_snrgnldf S_SNR_gnldf];
snrgnldf=[snrgnldf SNR_gnldf];
psnrgnldf=[psnrgnldf PSNR_gnldf];
betagnldf=[betagnldf BETA_gnldf];
fsimgnldf=[fsimgnldf FSIM_gnldf];
mssimgnldf=[mssimgnldf MSSIM_gnldf];

% GMF
rsgmf=[rsgmf RES_gmf];
cnrgmf=[cnrgmf CNR_gmf];
s_snrgmf=[s_snrgmf S_SNR_gmf];
snrgmf=[snrgmf SNR_gmf];
psnrgmf=[psnrgmf PSNR_gmf];
betagmf=[betagmf BETA_gmf];
fsimgmf=[fsimgmf FSIM_gmf];
mssimgmf=[mssimgmf MSSIM_gmf];

% KONGRES
rskong=[rskong RES_kong];
cnrkong=[cnrkong CNR_kong];
s_snrkong=[s_snrkong S_SNR_kong];
snrkong=[snrkong SNR_kong];
psnrkong=[psnrkong PSNR_kong];
betakong=[betakong BETA_kong];
fsimkong=[fsimkong FSIM_kong];
mssimkong=[mssimkong MSSIM_kong];

% SARSF
rssarsf=[rssarsf RES_sarsf];
cnrsarsf=[cnrsarsf CNR_sarsf];
s_snrsarsf=[s_snrsarsf S_SNR_sarsf];
snrsarsf=[snrsarsf SNR_sarsf];
psnrsarsf=[psnrsarsf PSNR_sarsf];
betasarsf=[betasarsf BETA_sarsf];
fsimsarsf=[fsimsarsf FSIM_sarsf];
mssimsarsf=[mssimsarsf MSSIM_sarsf];

end % end of runs


% Original Image
mrs0=abs(mean(rs0));
mcnr0=abs(mean(cnr0));
mssnr0=abs(mean(ssnr0));
msnr0=abs(mean(snr0));
mpsnr0=abs(mean(psnr0));
mbeta0=abs(mean(beta0));
mfsim0=abs(mean(fsim0));
mmssim0=abs(mean(mssim0));

Orig_img=[mrs0 mcnr0 mssnr0 msnr0 mpsnr0 mbeta0 mfsim0 mmssim0];

% PCA
mrspca=abs(mean(rspca));
mcnrpca=abs(mean(cnrpca));
ms_snr_pca=abs(mean(s_snrpca));
msnr_pca=abs(mean(snrpca));
mpsnr_pca=abs(mean(psnrpca));
mbeta_pca=abs(mean(betapca));
mfsim_pca=abs(mean(fsimpca));
mmssim_pca=abs(mean(mssimpca));

PCA=[mrspca, mcnrpca, ms_snr_pca, msnr_pca, mpsnr_pca, mbeta_pca,mfsim_pca, mmssim_pca];

% QR
mrsqr=abs(mean(rsqr));
mcnrqr=abs(mean(cnrqr));
ms_snr_qr=abs(mean(s_snrqr));
msnr_qr=abs(mean(snrqr));
mpsnr_qr=abs(mean(psnrqr));
mbeta_qr=abs(mean(betaqr));
mfsim_qr=abs(mean(fsimqr));
mmssim_qr=abs(mean(mssimqr));

QR=[mrsqr, mcnrqr, ms_snr_qr, msnr_qr, mpsnr_qr, mbeta_qr, mfsim_qr, mmssim_qr];

% SVD
mrssvd=abs(mean(rssvd));
mcnrsvd=abs(mean(cnrsvd));
ms_snr_svd=abs(mean(s_snrsvd));
msnr_svd=abs(mean(snrsvd));
mpsnr_svd=abs(mean(psnrsvd));
mbeta_svd=abs(mean(betasvd));
mfsim_svd=abs(mean(fsimsvd));
mmssim_svd=abs(mean(mssimsvd));

SVD=[mrssvd, mcnrsvd, ms_snr_svd, msnr_svd, mpsnr_svd, mbeta_svd, mfsim_svd, mmssim_svd];

% ARNOLDI
mrsarn=abs(mean(rsarn));
mcnrarn=abs(mean(cnrarn));
ms_snr_arn=abs(mean(s_snrarn));
msnr_arn=abs(mean(snrarn));
mpsnr_arn=abs(mean(psnrarn));
mbeta_arn=abs(mean(betaarn));
mfsim_arn=abs(mean(fsimarn));
mmssim_arn=abs(mean(mssimarn));

ARNOLDI=[mrsarn, mcnrarn, ms_snr_arn, msnr_arn, mpsnr_arn, mbeta_arn,mfsim_arn, mmssim_arn];


% LANCZOS
mrslancz=abs(mean(rslancz));
mcnrlancz=abs(mean(cnrlancz));
ms_snr_lancz=abs(mean(s_snrlancz));
msnr_lancz=abs(mean(snrlancz));
mpsnr_lancz=abs(mean(psnrlancz));
mbeta_lancz=abs(mean(betalancz));
mfsim_lancz=abs(mean(fsimlancz));
mmssim_lancz=abs(mean(mssimlancz));

LANCZ=[mrslancz, mcnrlancz, ms_snr_lancz, msnr_lancz, mpsnr_lancz, mbeta_lancz, mfsim_lancz, mmssim_lancz];


% ORTH
mrsorth=abs(mean(rsorth));
mcnrorth=abs(mean(cnrorth));
ms_snr_orth=abs(mean(s_snrorth));
msnr_orth=abs(mean(snrorth));
mpsnr_orth=abs(mean(psnrorth));
mbeta_orth=abs(mean(betaorth));
mfsim_orth=abs(mean(fsimorth));
mmssim_orth=abs(mean(mssimorth));

ORTHO=[mrsorth, mcnrorth, ms_snr_orth, msnr_orth, mpsnr_orth, mbeta_orth, mfsim_orth, mmssim_orth];

% SCHUR
mrsschur=abs(mean(rsschur));
mcnrschur=abs(mean(cnrschur));
ms_snr_schur=abs(mean(s_snrschur));
msnr_schur=abs(mean(snrschur));
mpsnr_schur=abs(mean(psnrschur));
mbeta_schur=abs(mean(betaschur));
mfsim_schur=abs(mean(fsimschur));
mmssim_schur=abs(mean(mssimschur));

SCHUR=[mrsschur, mcnrschur, ms_snr_schur, msnr_schur, mpsnr_schur, mbeta_schur, mfsim_schur, mmssim_schur];

% DWT
mrsdwt=abs(mean(rsdwt));
mcnrdwt=abs(mean(cnrdwt));
ms_snr_dwt=abs(mean(s_snrdwt));
msnr_dwt=abs(mean(snrdwt));
mpsnr_dwt=abs(mean(psnrdwt));
mbeta_dwt=abs(mean(betadwt));
mfsim_dwt=abs(mean(fsimdwt));
mmssim_dwt=abs(mean(mssimdwt));

DWT=[mrsdwt, mcnrdwt, ms_snr_dwt, msnr_dwt, mpsnr_dwt, mbeta_dwt, mfsim_dwt, mmssim_dwt];

% DWT2D
mrsdwt2D=abs(mean(rsdwt2D));
mcnrdwt2D=abs(mean(cnrdwt2D));
ms_snr_dwt2D=abs(mean(s_snrdwt2D));
msnr_dwt2D=abs(mean(snrdwt2D));
mpsnr_dwt2D=abs(mean(psnrdwt2D));
mbeta_dwt2D=abs(mean(betadwt2D));
mfsim_dwt2D=abs(mean(fsimdwt2D));
mmssim_dwt2D=abs(mean(mssimdwt2D));

DWT2D=[mrsdwt2D, mcnrdwt2D, ms_snr_dwt2D, msnr_dwt2D, mpsnr_dwt2D, mbeta_dwt2D, mfsim_dwt2D, mmssim_dwt2D];

% NLM
mrsnlm=abs(mean(rsnlm));
mcnrnlm=abs(mean(cnrnlm));
ms_snr_nlm=abs(mean(s_snrnlm));
msnr_nlm=abs(mean(snrnlm));
mpsnr_nlm=abs(mean(psnrnlm));
mbeta_nlm=abs(mean(betnlm));
mfsim_nlm=abs(mean(fsimnlm));
mmssim_nlm=abs(mean(mssimnlm));

NLM=[mrsnlm, mcnrnlm, ms_snr_nlm, msnr_nlm, mpsnr_nlm, mbeta_nlm, mfsim_nlm, mmssim_nlm];

% Wiener
mrswien=abs(mean(rswien));
mcnrwien=abs(mean(cnrwien));
ms_snr_wien=abs(mean(s_snrwien));
msnr_wien=abs(mean(snrwien));
mpsnr_wien=abs(mean(psnrwien));
mbeta_wien=abs(mean(betawien));
mfsim_wien=abs(mean(fsimwien));
mmssim_wien=abs(mean(mssimwien));

WIENER=[mrswien, mcnrwien, ms_snr_wien, msnr_wien, mpsnr_wien, mbeta_wien, mfsim_wien, mmssim_wien];

% PNLM
mrspnlm=abs(mean(rspnlm));
mcnrpnlm=abs(mean(cnrpnlm));
ms_snr_pnlm=abs(mean(s_snrpnlm));
msnr_pnlm=abs(mean(snrpnlm));
mpsnr_pnlm=abs(mean(psnrpnlm));
mbeta_pnlm=abs(mean(betapnlm));
mfsim_pnlm=abs(mean(fsimpnlm));
mmssim_pnlm=abs(mean(mssimpnlm));

PNLM=[mrspnlm, mcnrpnlm, ms_snr_pnlm, msnr_pnlm, mpsnr_pnlm, mbeta_pnlm, mfsim_pnlm, mmssim_pnlm];

% ADF
mrsadf=abs(mean(rsadf));
mcnradf=abs(mean(cnradf));
ms_snr_adf=abs(mean(s_snradf));
msnr_adf=abs(mean(snradf));
mpsnr_adf=abs(mean(psnradf));
mbeta_adf=abs(mean(betaadf));
mfsim_adf=abs(mean(fsimadf));
mmssim_adf=abs(mean(mssimadf));

ADF=[mrsadf, mcnradf, ms_snr_adf, msnr_adf, mpsnr_adf, mbeta_adf, mfsim_adf, mmssim_adf];

% TVF
mrstvf=abs(mean(rstvf));
mcnrtvf=abs(mean(cnrtvf));
ms_snr_tvf=abs(mean(s_snrtvf));
msnr_tvf=abs(mean(snrtvf));
mpsnr_tvf=abs(mean(psnrtvf));
mbeta_tvf=abs(mean(betatvf));
mfsim_tvf=abs(mean(fsimtvf));
mmssim_tvf=abs(mean(mssimtvf));

TVF=[mrstvf, mcnrtvf, ms_snr_tvf, msnr_tvf, mpsnr_tvf, mbeta_tvf, mfsim_tvf, mmssim_tvf];

% SRAD
mrssrad=abs(mean(rssrad));
mcnrsrad=abs(mean(cnrsrad));
ms_snr_srad=abs(mean(s_snrsrad));
msnr_srad=abs(mean(snrsrad));
mpsnr_srad=abs(mean(psnrsrad));
mbeta_srad=abs(mean(betasrad));
mfsim_srad=abs(mean(fsimsrad));
mmssim_srad=abs(mean(mssimsrad));

SRAD=[mrssrad, mcnrsrad, ms_snr_srad, msnr_srad, mpsnr_srad, mbeta_srad, mfsim_srad, mmssim_srad];

% FROST
mrsfrost=abs(mean(rsfrost));
mcnrfrost=abs(mean(cnrfrost));
ms_snr_frost=abs(mean(s_snrfrost));
msnr_frost=abs(mean(snrfrost));
mpsnr_frost=abs(mean(psnrfrost));
mbeta_frost=abs(mean(betafrost));
mfsim_frost=abs(mean(fsimfrost));
mmssim_frost=abs(mean(mssimfrost));

FROST=[mrsfrost, mcnrfrost, ms_snr_frost, msnr_frost, mpsnr_frost, mbeta_frost, mfsim_frost, mmssim_frost];

% KUAN
mrskuan=abs(mean(rskuan));
mcnrkuan=abs(mean(cnrkuan));
ms_snr_kuan=abs(mean(s_snrkuan));
msnr_kuan=abs(mean(snrkuan));
mpsnr_kuan=abs(mean(psnrkuan));
mbeta_kuan=abs(mean(betakuan));
mfsim_kuan=abs(mean(fsimkuan));
mmssim_kuan=abs(mean(mssimkuan));

KUAN=[mrskuan, mcnrkuan, ms_snr_kuan, msnr_kuan, mpsnr_kuan, mbeta_kuan, mfsim_kuan, mmssim_kuan];

% LEE
mrslee=abs(mean(rslee));
mcnrlee=abs(mean(cnrlee));
ms_snr_lee=abs(mean(s_snrlee));
msnr_lee=abs(mean(snrlee));
mpsnr_lee=abs(mean(psnrlee));
mbeta_lee=abs(mean(betalee));
mfsim_lee=abs(mean(fsimlee));
mmssim_lee=abs(mean(mssimlee));

LEE=[mrslee, mcnrlee, ms_snr_lee, msnr_lee, mpsnr_lee, mbeta_lee, mfsim_lee, mmssim_lee];

% NCDF
mrsncdf=abs(mean(rsncdf));
mcnrncdf=abs(mean(cnrncdf));
ms_snr_ncdf=abs(mean(s_snrncdf));
msnr_ncdf=abs(mean(snrncdf));
mpsnr_ncdf=abs(mean(psnrncdf));
mbeta_ncdf=abs(mean(betancdf));
mfsim_ncdf=abs(mean(fsimncdf));
mmssim_ncdf=abs(mean(mssimncdf));

NCDF=[mrsncdf, mcnrncdf, ms_snr_ncdf, msnr_ncdf, mpsnr_ncdf, mbeta_ncdf, mfsim_ncdf, mmssim_ncdf];

% BM3D
mrsbm3d=abs(mean(rsbm3d));
mcnrbm3d=abs(mean(cnrbm3d));
ms_snr_bm3d=abs(mean(s_snrbm3d));
msnr_bm3d=abs(mean(snrbm3d));
mpsnr_bm3d=abs(mean(psnrbm3d));
mbeta_bm3d=abs(mean(betabm3d));
mfsim_bm3d=abs(mean(fsimbm3d));
mmssim_bm3d=abs(mean(mssimbm3d));


BM3D=[mrsbm3d, mcnrbm3d, ms_snr_bm3d, msnr_bm3d, mpsnr_bm3d, mbeta_bm3d, mfsim_bm3d, mmssim_bm3d];

% OBNLM
mrsobnlm=abs(mean(rsobnlm));
mcnrobnlm=abs(mean(cnrobnlm));
ms_snr_obnlm=abs(mean(s_snrobnlm));
msnr_obnlm=abs(mean(snrobnlm));
mpsnr_obnlm=abs(mean(psnrobnlm));
mbeta_obnlm=abs(mean(betaobnlm));
mfsim_obnlm=abs(mean(fsimobnlm));
mmssim_obnlm=abs(mean(mssimobnlm));


OBNLM=[mrsobnlm, mcnrobnlm, ms_snr_obnlm, msnr_obnlm, mpsnr_obnlm, mbeta_obnlm, mfsim_obnlm, mmssim_obnlm];

% DPAD
mrsdpad=abs(mean(rsdpad));
mcnrdpad=abs(mean(cnrdpad));
ms_snr_dpad=abs(mean(s_snrdpad));
msnr_dpad=abs(mean(snrdpad));
mpsnr_dpad=abs(mean(psnrdpad));
mbeta_dpad=abs(mean(betadpad));
mfsim_dpad=abs(mean(fsimdpad));
mmssim_dpad=abs(mean(mssimdpad));

DPAD=[mrsdpad, mcnrdpad, ms_snr_dpad, msnr_dpad, mpsnr_dpad, mbeta_dpad, mfsim_dpad, mmssim_dpad];

% GAMMA
mrsgamma=abs(mean(rsgamma));
mcnrgamma=abs(mean(cnrgamma));
ms_snr_gamma=abs(mean(s_snrgamma));
msnr_gamma=abs(mean(snrgamma));
mpsnr_gamma=abs(mean(psnrgamma));
mbeta_gamma=abs(mean(betagamma));
mfsim_gamma=abs(mean(fsimgamma));
mmssim_gamma=abs(mean(mssimgamma));

GAMMA=[mrsgamma, mcnrgamma, ms_snr_gamma, msnr_gamma, mpsnr_gamma, mbeta_gamma, mfsim_gamma, mmssim_gamma];

% GNLDF
mrsgnldf=abs(mean(rsgnldf));
mcnrgnldf=abs(mean(cnrgnldf));
ms_snr_gnldf=abs(mean(s_snrgnldf));
msnr_gnldf=abs(mean(snrgnldf));
mpsnr_gnldf=abs(mean(psnrgnldf));
mbeta_gnldf=abs(mean(betagnldf));
mfsim_gnldf=abs(mean(fsimgnldf));
mmssim_gnldf=abs(mean(mssimgnldf));

GNLDF=[mrsgnldf, mcnrgnldf, ms_snr_gnldf, msnr_gnldf, mpsnr_gnldf, mbeta_gnldf, mfsim_gnldf, mmssim_gnldf];

% GMF
mrsgmf=abs(mean(rsgmf));
mcnrgmf=abs(mean(cnrgmf));
ms_snr_gmf=abs(mean(s_snrgmf));
msnr_gmf=abs(mean(snrgmf));
mpsnr_gmf=abs(mean(psnrgmf));
mbeta_gmf=abs(mean(betagmf));
mfsim_gmf=abs(mean(fsimgmf));
mmssim_gmf=abs(mean(mssimgmf));

GMF=[mrsgmf, mcnrgmf, ms_snr_gmf, msnr_gmf, mpsnr_gmf, mbeta_gmf, mfsim_gmf, mmssim_gmf];

% KONGRES
mrskong=abs(mean(rskong));
mcnrkong=abs(mean(cnrkong));
ms_snr_kong=abs(mean(s_snrkong));
msnr_kong=abs(mean(snrkong));
mpsnr_kong=abs(mean(psnrkong));
mbeta_kong=abs(mean(betakong));
mfsim_kong=abs(mean(fsimkong));
mmssim_kong=abs(mean(mssimkong));

KONGRES=[mrskong, mcnrkong, ms_snr_kong, msnr_kong, mpsnr_kong, mbeta_kong, mfsim_kong, mmssim_kong];

% SARSF
mrssarsf=abs(mean(rssarsf));
mcnrsarsf=abs(mean(cnrsarsf));
ms_snr_sarsf=abs(mean(s_snrsarsf));
msnr_sarsf=abs(mean(snrsarsf));
mpsnr_sarsf=abs(mean(psnrsarsf));
mbeta_sarsf=abs(mean(betasarsf));
mfsim_sarsf=abs(mean(fsimsarsf));
mmssim_sarsf=abs(mean(mssimsarsf));

SARSF=[mrssarsf, mcnrsarsf, ms_snr_sarsf, msnr_sarsf, mpsnr_sarsf, mbeta_sarsf, mfsim_sarsf, mmssim_sarsf];


NUM_RESULTS=[Orig_img;PCA;QR;SVD;ARNOLDI;LANCZ;ORTHO;SCHUR;DWT;DWT2D;NLM;WIENER;PNLM;ADF;TVF;SRAD;FROST;KUAN;LEE;NCDF;BM3D;OBNLM;DPAD;GAMMA;GNLDF;GMF;KONGRES;SARSF];
VIS_RESULTS=[dn, IM, estespca, estesqr, estessvd, estesarn,esteslancz,estesorth,estesschur,estesdwt,estesdwt2D,estesnlm, esteswien,estespnlm, estesadf, estestvf, estessrad, estesfrost, esteskuan, esteslee, estesncdf, estesbm3d, estesobnlm,estesdpad,estesgamma,estesgnldf,estesgmf,esteskong,estessarsf];
schemes={'Orig---'; 'PCA----'; 'QR-----'; 'SVD----'; 'Arnoldi'; 'lanczos';'Ortho--';'Schur--';'DWT----'; 'DWT2D--';'NLM----'; 'Wiener-';'PNLM---';'ADF----'; 'TVF----';  'SRAD---'; 'Frost--'; 'Kuan---'; 'Lee----'; 'NCDF---'; 'BM3D---'; 'OBNLM--';'DPAD---';'Gamma--';'GNLDF--';'GMF----';'KONGRES';'SARSF--'};

parameters={'Methods','RES----','CNR----', 'S_SNR--', 'SNR----', 'PSNR---', 'BETA---', 'FSIM---', 'MSSIM--'};

schemRes=[schemes num2cell(NUM_RESULTS)];

NumResults = [parameters;schemRes]


%save Disc_Rough_GRAPHS.mat NF IM estespca estesqr estessvd estesarn esteslancz estespnlm estesfrost esteslee estesdpad estesgnldf Orig_img QR SVD ARNOLDI LANCZ PNLM FROST LEE DPAD GNLDF PCA

% save Disc_Lanc_SVD_PNLM_FROST_Lee_DPAD_GNDF.mat NF IM estesqr estessvd estesarn esteslancz estespnlm estesfrost esteslee estesdpad estesgnldf Orig_img QR SVD ARNOLDI LANCZ PNLM FROST LEE DPAD GNLDF

% dnR=dn;
% IMR=IM;
% estesqrR=estesqr;
% estessvdR=estessvd;
% estesarnR=estesarn;
% esteslanczR=esteslancz;
% estespnlmR=estespnlm;
% estesfrostR=estesfrost;
% estesleeR=esteslee;
% estesgnldfR=estesgnldf;




% save Disc_Lanc_SVD_PNLM_FROST_Lee_DPAD_GNDF.mat NF IM estesqr estessvd estesarn esteslancz estespnlm estesfrost esteslee estesdpad estesgnldf Orig_img QR SVD ARNOLDI LANCZ PNLM FROST LEE DPAD GNLDF


% save MRI_Analysis.mat estespca estessvd estesarn estesdwt estesnlm esteswien PCA SVD ARNOLDI DWT NLM WIENER

% log_env1=IM;
% log_env=log(log_env1+0.01);
% log_env=log_env-min(min(log_env));
% log_env=64*log_env/max(max(log_env));
% image(log_env)
% colormap(gray(64))
% brighten(0.2)

estesgnldf = double(estesgnldf);  % its class was single

svdorthdiff = imabsdiff(dn,estesqr);
gsvdoblidiff = imabsdiff(dn,estessvd);
lanczorthdiff = imabsdiff(dn,estesarn);
lanczoblidiff = imabsdiff(dn,esteslancz);
pnlmdiff = imabsdiff(dn,estespnlm);
frostdiff = imabsdiff(dn,estesfrost);
leediff = imabsdiff(dn,esteslee);
gnldfdiff = imabsdiff(dn,estesgnldf);
gsrbfdiff = imabsdiff(dn,estespca);

Ssvdorth=sum(svdorthdiff(:))/sum(dn(:))
Ssvdobli=sum(gsvdoblidiff(:))/sum(dn(:))
SLancorth=sum(lanczorthdiff(:))/sum(dn(:))
SLancobli=sum(lanczoblidiff(:))/sum(dn(:))
Spnlm=sum(pnlmdiff(:))/sum(dn(:))
Sfrost=sum(frostdiff(:))/sum(dn(:))
SLee=sum(leediff(:))/sum(dn(:))
Sgndlf=sum(gnldfdiff(:))/sum(dn(:))
Sgsrbf=sum(gsrbfdiff(:))/sum(dn(:))

figure(1)

subplot(2,2,1)
imshow(svdorthdiff)
colormap(gray(64))
xlabel('[a]')
title('SVD ORTH')

subplot(2,2,2)
imshow(gsvdoblidiff)
colormap(gray(64))
xlabel('[b]')
title('SVD OBLI')

subplot(2,2,3)
imshow(lanczorthdiff)
colormap(gray(64))
xlabel('[c]')
title('LANCZOS ORTH')

subplot(2,2,4)
imshow(lanczoblidiff)
colormap(gray(64))
xlabel('[d]')
title('LANCZOS OBLI')

figure(2)

subplot(2,2,1)
imshow(pnlmdiff)
colormap(gray(64))
xlabel('[e]')
title('PNLM')

subplot(2,2,2)
imshow(frostdiff)
colormap(gray(64))
xlabel('[f]')
title('FROST')

subplot(2,2,3)
imshow(leediff)
colormap(gray(64))
xlabel('[g]')
title('LEE')

subplot(2,2,4)
imshow(gnldfdiff)
colormap(gray(64))
xlabel('[h]')
title('GNLDF')

figure(3)

subplot(2,2,1)
imshow(gbfdiff)
colormap(gray(64))
xlabel('[i]')
title('GSR')

subplot(2,2,2)
imshow(frostdiff)
colormap(gray(64))
xlabel('[f]')
title('FROST')

subplot(2,2,3)
imshow(leediff)
colormap(gray(64))
xlabel('[g]')
title('LEE')

subplot(2,2,4)
imshow(gnldfdiff)
colormap(gray(64))
xlabel('[h]')
title('GNLDF')

% % figure(1)
% % 
% %  subplot(1,2,1)
% %  
% % log_env=log(dn+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env1=64*log_env/max(max(log_env));
% % image(log_env1)
% % colormap(gray(64))
% %  axis off
% % title('Reference')
% % xlabel('[a]')
% %  
% %  
% %  
% %  subplot(1,2,2)
% % 
% % log_env=log(IM+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env2=64*log_env/max(max(log_env));
% % image(log_env2)
% % colormap(gray(64))
% % axis off
% % title('Speckle Noisy')
% % xlabel('[b]')
% %  
% %  figure(2)
% %  
% %  subplot(1,2,1)
% %  
% %  log_env=log(estespca+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env2=64*log_env/max(max(log_env));
% % image(log_env2)
% % colormap(gray(64))
% % axis off
% % title('PCA')
% %  
% %  
% %  subplot(1,2,2)
% % 
% %  log_env=log(estesqr+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env3=64*log_env/max(max(log_env));
% % image(log_env3)
% % colormap(gray(64))
% % axis off
% % title('QR ORTH')
% % xlabel('[e]')
% %  
% %  
% %  figure(3)
% % 
% %  subplot(1,2,1)
% % 
% %   log_env=log(estessvd+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env4=64*log_env/max(max(log_env));
% % image(log_env4)
% % colormap(gray(64))
% % axis off
% % title('QR OBLI')
% % xlabel('[f]')
% %  
% %  
% %  subplot(1,2,2)
% %  
% % log_env=log(estesarn+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env5=64*log_env/max(max(log_env));
% % image(log_env5)
% % colormap(gray(64))
% % axis off
% % title('LANCZOS ORTH')
% % xlabel('[c]')
% %  
% %  figure (4)
% %  
% %   subplot(1,2,1)
% %  imshow(estesorth)
% %  brighten(.1)
% %  title('ORTHONORMAL')
% %  
% %  subplot(1,2,2)
% %  
% % log_env=log(esteslancz+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env6=64*log_env/max(max(log_env));
% % image(log_env6)
% % colormap(gray(64))
% % axis off
% % title('LANCZOS OBLI')
% % xlabel('[d]')
% %  
% %  
% %  figure(5)
% %  
% %  subplot(1,2,1)
% %  imshow(estesschur)
% %  brighten(.1)
% %  title('SCHUR')
% %  
% %  subplot(1,2,2)
% %  imshow(estesdwt)
% %  brighten(.1)
% %  title('DWT')
% %  
% %  figure(6)
% %  
% %  subplot(1,2,1)
% %  imshow(estesdwt2D)
% %  brighten(.1)
% %  title('DWT2D')
% %  
% %  subplot(1,2,2)
% %  imshow(estesnlm)
% %  brighten(.1)
% %  title('NLM')
% %  
% %  
% %  figure(7)
% %  
% %  subplot(1,2,1)
% %  imshow(esteswien)
% %  brighten(.1)
% %  title('Wiener')
% %  
% %  subplot(1,2,2)
% %  
% %   log_env=log(estespnlm+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env7=64*log_env/max(max(log_env));
% % image(log_env7)
% % colormap(gray(64))
% % axis off
% %  title('PNLM')
% % xlabel('[g]')
% % 
% %  figure(8)
% %  subplot(1,2,1)
% %  imshow(estesadf)
% %  brighten(.1)
% %  title('ADF')
% % 
% %  subplot(1,2,2)
% %  imshow(estestvf)
% %  brighten(.1)
% %  title('TVF')
% %   
% %  figure(9)
% %  
% %  subplot(1,2,1)
% %  imshow(estessrad)
% %  brighten(.1)
% %  title('SRAD')
% %  
% %  subplot(1,2,2)
% % 
% %    log_env=log(estesfrost+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env8=64*log_env/max(max(log_env));
% % image(log_env8)
% % colormap(gray(64))
% % axis off
% %   title('FROST')
% % xlabel('[h]')
% % 
% %  figure(10)
% %  subplot(1,2,1)
% %  imshow(esteskuan)
% %  brighten(.1)
% %  title('Kuan')
% %  
% %  subplot(1,2,2)
% %  
% % log_env=log(esteslee+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env9=64*log_env/max(max(log_env));
% % image(log_env9)
% % colormap(gray(64))
% % axis off
% %   title('LEE')
% % xlabel('[i]')
% %  
% %  
% %  
% %   figure(11)
% % 
% %  subplot(1,2,1)
% %  imshow(estesncdf)
% %  brighten(.1)
% %  title('NCDF')
% %   
% %  subplot(1,2,2)
% %  imshow(estesbm3d)
% %  brighten(.1)
% %  title('BM3D')
% %  
% %  figure(12)
% %  
% %  subplot(1,2,1)
% %  imshow(estesobnlm)
% %  brighten(.1)
% %  title('OBNLM')
% %  
% %  subplot(1,2,2)
% % 
% % log_env=log(estesdpad+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env10=64*log_env/max(max(log_env));
% % image(log_env10)
% % colormap(gray(64))
% % axis off
% %   title('DPAD')
% % xlabel('[j]')
% % 
% %  
% %    figure(13)
% % 
% %  subplot(1,2,1)
% %  imshow(estesgamma)
% %  brighten(.1)
% %  title('GAMMA')
% %   
% %  subplot(1,2,2)
% %  imshow(estesgnldf)
% %  log_env=log(estesgnldf+0.01);
% % log_env=log_env-min(min(log_env));
% % log_env11=64*log_env/max(max(log_env));
% % image(log_env11)
% % colormap(gray(64))
% % axis off
% %   title('GNLDF')
% % xlabel('[k]')

% figure(13)
%  
%  subplot(1,2,1)
%  imshow(estesgmf)
%  brighten(.1)
%  title('GMF')
%  
%  subplot(1,2,2)
%  imshow(esteskong)
%  brighten(.1)
%  title('KONGRES')
%  
%  figure(14)
% 
%  subplot(1,2,1)
%  imshow(estessarsf)
%  brighten(.1)
%  title('SARSF')
%   
%  subplot(1,2,2)
%  imshow(estesgnldf)
%  brighten(.1)
%  title('GNLDF')
%  
%  figure(15)
%  subplot(2,2,3)
%  imshow(estesgmf)
%  brighten(.1)
%  title('GMF')
%  
%  subplot(2,2,4)
%  imshow(esteskong)
%  brighten(.1)
%  title('KONGRES')

 
