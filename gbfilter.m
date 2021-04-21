
function [yb] =  gbfilter(A, n)

win=[n n]; 
ksize=n;

sigmas=10;
sigmar=10;

Ismooth = imguidedfilter(A,'NeighborhoodSize',win);

%logIsmooth=log(Ismooth);

[yb, ~]=bilateral_filt2D(Ismooth,sigmas,sigmar,ksize);

%yb= exp(yb);



