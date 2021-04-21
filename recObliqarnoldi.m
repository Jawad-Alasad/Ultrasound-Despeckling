function denArn=recObliqarnoldi(D2,S1,S2,pow)


ms1=S1;        
ms2=S2;
blockspace=zeros(ms1*ms2);
%zt=NonLinear_soft(S1*S2,pow);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nr mc]=size(D2);

noisy=D2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nb=0;

for r=1:nr-ms1+1
    
for c=1:mc-ms2+1           
    
    nb=nb+1;
    
 win=noisy(r:r+ms1-1,c:c+ms2-1);        
      
 Rn=reshape(win,ms1*ms2,1);
 
 covs=Rn*Rn';  
 
 blockspace=blockspace+covs;
         
end
end

cov=blockspace/nb;

k=length(cov);
% cov here is the global covariance matrix for the given image (msxms)

[Q,H] = arnoldi(cov,k);  % V eigvect and D eigval


% % % % [Q,H]=lansym(cov,k);
% % % % [Q,D] = eig(cov);
[Q H] =QR_sort_VEC_VAL_B2S(Q, H);
% % % %  P=B1*pinv(B2p'*B1)*B2p';  % Oblique Projection

Id=eye(k);

B1=Q(:,1:pow);

B2=Q(:,k-pow+1:k);

%B2=Q(:,k);

%B2=fliplr(B2);


% PH=B1*pinv(B1'*B1)*B1';
PH=B1*((B1'*B1)\B1')   ;     % Orthogonal    

% PHm=min(PH(:));   %%%% Added
%  
% PH=PH+PHm;

PHC=Id-PH;


% P=PH*(Id-B2*pinv(B2'*PHC*B2)*B2'*PHC);
P=PH*(Id-B2*((B2'*PHC*B2)\B2')*PHC);   % Oblique


% % % %B2=1./Q(:,1:pow);
% % % 
% % % %B2=Q(:,k-pow+1-40:k-40);
% % %  
% % % % B2est=B2*ones(k-pow,k)-0.2*B1*ones(pow,k);
% % % % PNorth = Id - pinv(B2est'*B2est)*B2est'; 
% % % 
% % % %%%%%%%%%%%%%%%%%%  PN=B2*pinv(B2'*B2)*B2';
% % % 
% % % %PNorth=Id-B2*B2';
% % % PNorth=Id-B2*B2';
% % % 
% % % P=B1*pinv(B1'*PNorth*B1)*B1'*PNorth;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

denspace1=zeros(nr,mc);
denspace2=zeros(nr,mc);

 win0=ones(ms1,ms2);

for r=1:nr-ms1+1
    
for c=1:mc-ms2+1           
    
       
 win=noisy(r:r+ms1-1,c:c+ms2-1);          
      
 Rd=reshape(win,ms1*ms2,1);
 
 
 DNS=P*Rd;  %%%%%%%%%%%%%%%%%%%%?
 
 rdns=reshape(DNS,ms1,ms2);

 
denspace1(r:r+ms1-1,c:c+ms2-1)= denspace1(r:r+ms1-1,c:c+ms2-1)+ rdns;
denspace2(r:r+ms1-1,c:c+ms2-1)= denspace2(r:r+ms1-1,c:c+ms2-1)+ win0;
         
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

denArn=(denspace1./denspace2);

    