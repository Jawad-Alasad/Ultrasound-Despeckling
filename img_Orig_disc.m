

function dn=img_Orig_disc(env, D, sz)

 
 [en em]=size(env);

Zplane=zeros(en,em*sz);

r=0;
for i=1:1/sz:em
    
    r=r+1;
    
       Zplane(:,r)=env(:,i);
end

env=Zplane;

%D=16;   %  Sampling frequency decimation factor
 
 dn=env(1:D:max(size(env)),:)/max(max(env));