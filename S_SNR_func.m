
function S_SNR=S_SNR_func(x)

% x: estimated image
% S_SNR: Speckle signal to noise ratio

S_SNR=mean(x(:))/std(x(:));