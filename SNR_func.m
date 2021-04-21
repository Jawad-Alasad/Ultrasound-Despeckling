
function SNR=SNR_func(dn,x)

% SNR:  Signal to noise ratio.
% dn:   Reference Image 
% x: estimated image

SNR=20*log10(norm(dn)/norm(dn-x));