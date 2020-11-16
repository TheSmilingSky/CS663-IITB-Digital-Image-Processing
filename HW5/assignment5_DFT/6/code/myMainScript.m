%% MyMainScript

tic;
k1 = [0,1,0; 1,-4,1; 0,1,0];
k2 = [-1,-1,-1; -1,8,-1; -1,-1,-1];

Fk1 = fftshift(fft2(k1,201,201));
Fk2 = fftshift(fft2(k2,201,201));
lf_k1 = log(abs(Fk1)+1);
lf_k2 = log(abs(Fk2)+1);
fig1 = figure(1);
subplot(121); imshow(lf_k1,[-1 2]); colormap(jet); colorbar; 
title('N,N-point DFT of k1');
subplot(122); imshow(lf_k2,[-1 2]); colormap(jet); colorbar;
title('N,N-point DFT of k2');
toc;
