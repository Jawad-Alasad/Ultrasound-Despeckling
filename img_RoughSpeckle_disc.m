
%img_Speckle_disc

function  IM=img_RoughSpeckle_disc(i,D,sz)

cmd=['load DiskNoisy',num2str(i),'.mat'];  %N1Reading
  eval(cmd)
  
%D=16;   %  Sampling frequency decimation factor

[en em]=size(envN);

Zplane=zeros(en,em*sz);

r=0;
for i=1:1/sz:em
    
    r=r+1;
    
       Zplane(:,r)=envN(:,i);
end

envo=Zplane;


est0=envo;

IM=est0(1:D:max(size(est0)),:)/max(max(est0));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

