function f = Lee_speckle1_removal(g, nhood, niterations)
%
%SPECKLE REDUCTION  FILTER. Christos Loizou 2001

% J.S. Lee, Speckle analysis and smoothing of synthetic aperture radar
% images, Computer graphics and image processing, n. 17, 1981, pp 24-32

%Best Results by K=specklel (I, [3 3], 3);

%Calculate 56 Texture Features of the original (unfiltered) Image
%A=[ ]; F1=[ ] ;         % Initialize the matrcies for texture features
%T= texfeat(double(g));  % Calculate Texture features for the original image
%A=[A, T'];
%save or_texfeats A;     %Save the texture Features in a matrix

if isa(g, 'uint8')
  u8out = 1;
  if (islogical(g))
    % It doesn't make much sense to pass a binary image
    % in to this function, but just in case.
    logicalOut = 1;
    g = double(g);
else
    logicalOut = 0;  
    g = double(g)/255;    
end
else
  u8out = 0;
end

%Calculate the noise and the noise variance in the image 
%noise=noisevar(z, nhood, ma, na, g);
stdnoise=(std2(g).*std2(g))/mean2(g);
noisevar=stdnoise*stdnoise; %noise variance 

%Initialize the picture f (new picture) with zeros
f = g;

for i = 1:niterations           %Apply niteration of the algorithm to the image 
  % fprintf('\rIteration %d',i);
  if i >=2 
      g=f;
  end
  
%Estimate the local mean of f.
localMean = filter2(ones(nhood), g) / prod(nhood);
lmsqr = localMean.*localMean;       % square of the local mean

%Estimate of the local variance of f.
localVar = filter2(ones(nhood), g.^2) / prod(nhood) - localMean.^2;

% Estimate the noise power in the image
%noiseVar = mean2(localVar);

%Compute new image f from noise image g
f=localMean + (localVar - lmsqr .*noisevar ./ ...
				 max(0.1, localVar + lmsqr .* noisevar)) .* (g - localMean); 

end %end for i Itterations 
% fprintf('\n');


% if u8out==1,
%   if (logicalOut)
%     f = uint8(f);
% else
%     f = uint8(round(f*255));
% end
% end


%f=f.*255;

%Calculate 56 texture features for the filtered image
%TAM=texfeat(double(f));
%F1=[F1,TAM'];
%save speckle1texfs F1;

% figure, imshow(f);
% title('Image filtered by speckle1 filter');

%for saving the image directly into an image file 
%imwrite(f, 'speckle5[7 7].bmp', 'bmp');