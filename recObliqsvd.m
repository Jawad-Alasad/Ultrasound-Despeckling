

function denSVD=recObliqsvd(D2,S1,S2,pow)
 

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


 [V,D,R] = svd(cov); % SVD decomposition

 [Q H] =QR_sort_VEC_VAL_B2S(V, D);
 
Id=eye(k);

B1=Q(:,1:pow);

B2=Q(:,k-pow+1:k);


%B2=fliplr(B2);

% PH=B1*pinv(B1'*B1)*B1';
PH=B1*((B1'*B1)\B1');   %+1/(S1*S2)


PHC=Id-PH;


% P=PH*(Id-B2*pinv(B2'*PHC*B2)*B2'*PHC);
P=PH*(Id-B2*((B2'*PHC*B2)\B2')*PHC);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

denspace1=zeros(nr,mc);
denspace2=zeros(nr,mc);

 win0=ones(ms1,ms2);

for r=1:nr-ms1+1
    
for c=1:mc-ms2+1           
    
       
 win=noisy(r:r+ms1-1,c:c+ms2-1);          
      
 Rd=reshape(win,ms1*ms2,1);
 
 DNS=P*Rd;  
 
 rdns=reshape(DNS,ms1,ms2);
 
 
denspace1(r:r+ms1-1,c:c+ms2-1)= denspace1(r:r+ms1-1,c:c+ms2-1)+ rdns;
denspace2(r:r+ms1-1,c:c+ms2-1)= denspace2(r:r+ms1-1,c:c+ms2-1)+ win0;
         
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

denSVD=(denspace1./denspace2);

