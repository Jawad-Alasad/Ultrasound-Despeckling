
function RES = RES_func(IM)

% IM: estimated image
% RES: resolution of estimated image; the lower the better

imr=IM;

sigma2_x = var(imr(:));
mean_x = mean(imr(:));
imr_r = circshift(imr,[1 0]);
imr_c = circshift(imr,[0 1]);
rho_mat = corrcoef([imr(:); imr(:)],[imr_r(:); imr_c(:)]);
rho = rho_mat(1,2);
[rr,cc] = ndgrid([-64:63],[-32:31]);
RC0 = sigma2_x*rho.^sqrt(rr.^2+cc.^2) + mean_x^2;

RC0m=RC0/max(RC0(:));
RC0m=RC0m-min(RC0m(:)); 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
[nc0 mc0]=size(RC0);

CRC0=0.75*max(RC0m(:));  % axceeds 0.75 of its max value

for i=1:nc0
    for j=1:mc0
        if RC0m(i,j)>CRC0;
            R0(i,j)=1;
        else
            R0(i,j)=0;
        end
    end
end

RES=sum(R0(:))/(nc0*mc0);
