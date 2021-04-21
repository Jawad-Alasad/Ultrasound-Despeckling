
function PSNR = PSNR_func(dn,x)

% PSNR: Peak Signal to noise ratio.
% dn:   Reference Image 
% x: estimated image

%DHP=dn;  % Pure image 
maxDM=max(dn(:));  %maxDM should be for the noise free image see(http://bigwww.epfl.ch/preprints/luisier0602p.pdf)

%PCA
DMPw=x;  
DM2w=(dn-DMPw).^2;
MSEw=mean(DM2w(:));
PSNR=10*log10(maxDM^2/MSEw);