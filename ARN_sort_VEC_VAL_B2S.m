
function [N_EVEC N_EVAL] =ARN_sort_VEC_VAL_B2S(EigVect, EigVal)


EigVal=eye(size(EigVal)).*EigVal;

SJ=abs(sum(EigVal)); 

%%%%%% SJ=abs(sum(EigVal));  % fundamental step

[S I]=sort(SJ,'descend'); %  'descend'. 'ascend' for arnoldi

L=length(I);

Zmat=zeros(L);

for i=1:L
    
    N_EVEC(:,i)=EigVect(:,I(i));
    T_EVAL(:,i)=EigVal(:,I(i));
   
end

N_EVEC;
T_EVAL;

for i=1:L
    
    ST_EVAL=sum(T_EVAL(:,i));
    
    Zmat(i,i)=ST_EVAL;

end

N_EVAL=Zmat;

