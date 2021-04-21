

function denSVD=recOrthsvd(D2,S1,S2,pow)
 

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

% cov here is the global covariance matrix for the given image (msxms)


 [U,T,F] = svd(cov); % SVD decomposition

 [Q H] =QR_sort_VEC_VAL_B2S(U, T);

% U=Q(:,1:pow); % R or V

B1=Q(:,1:pow);
PH=B1*((B1'*B1)\B1');     % Orthogonal +1/(S1*S2)

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Finding P  transformation or projection matrix
%P=U*U';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

denspace1=zeros(nr,mc);
denspace2=zeros(nr,mc);

 win0=ones(ms1,ms2);

for r=1:nr-ms1+1
    
for c=1:mc-ms2+1           
    
       
 win=noisy(r:r+ms1-1,c:c+ms2-1);          
      
 Rd=reshape(win,ms1*ms2,1);
 
 DNS=PH*Rd;  
 
 rdns=reshape(DNS,ms1,ms2);
 

denspace1(r:r+ms1-1,c:c+ms2-1)= denspace1(r:r+ms1-1,c:c+ms2-1)+ rdns;
denspace2(r:r+ms1-1,c:c+ms2-1)= denspace2(r:r+ms1-1,c:c+ms2-1)+ win0;
         
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

denSVD=(denspace1./denspace2);

