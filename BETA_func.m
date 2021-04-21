
function BETA = BETA_func(dn,x)

% Beta: edge detection, the closer to one the better.
% dn:   Reference Image 
% x: estimated image

Iorg=dn; %  Reference Image

LA_Iorgcm = edge(Iorg,'log');
LA_IorgDcm = im2double(LA_Iorgcm);
Iorg_normcm=norm(LA_IorgDcm,'fro');


% PCA
Iepca=x; %
LA_Iestwv = edge(Iepca,'log');
LA_IestDw = im2double(LA_Iestwv);
IorgIestwv=LA_IorgDcm.*LA_IestDw;
Iest_normw=norm(LA_IestDw,'fro');
Betaw=sum(IorgIestwv(:))/(Iorg_normcm.*Iest_normw);
BETA=mean(Betaw(:));