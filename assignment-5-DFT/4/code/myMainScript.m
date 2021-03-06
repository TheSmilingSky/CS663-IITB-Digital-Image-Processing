%% MyMainScript

tic;
%% Your code here
img = imread('../data/barbara256.png');
[m,n] = size(img);
img = padarray(img,[m/2,n/2],0);
fimg = fftshift(fft2(img));
figure;
imshow(mat2gray(log(1+abs(fimg))));
colormap jet;
colorbar;

vars = [40,80];
H11 = zeros(2*m,2*n);
H12 = zeros(2*m,2*n);
H21 = zeros(2*m,2*n);
H22 = zeros(2*m,2*n);
for i=1:2*m
    for j=1:2*n
        if ((i-m)^2 + (j-n)^2 <= vars(1)^2)
            H11(i,j) = 1;
        end
        if ((i-m)^2 + (j-n)^2 <= vars(2)^2)
            H12(i,j) = 1;
        end
        H21(i,j) = exp(-((i-m)^2 + (j-n)^2)/(2*vars(1)^2));
        H22(i,j) = exp(-((i-m)^2 + (j-n)^2)/(2*vars(2)^2));
    end
end

img11 = ifft2(ifftshift(fimg.*H11));
img11 = img11(m/2+1:3*m/2,n/2+1:3*n/2);
figure;
imshow(real(img11),[min(abs(img11(:))) max(abs(img11(:)))]);

img12 = ifft2(ifftshift(fimg.*H12));
img12 = img12(m/2+1:3*m/2,n/2+1:3*n/2);
figure;
imshow(real(img12),[min(abs(img12(:))) max(abs(img12(:)))]);
toc;

img21 = ifft2(ifftshift(fimg.*H21));
img21 = img21(m/2+1:3*m/2,n/2+1:3*n/2);
figure;
imshow(real(img21),[min(abs(img21(:))) max(abs(img21(:)))]);

img22 = ifft2(ifftshift(fimg.*H22));
img22 = img22(m/2+1:3*m/2,n/2+1:3*n/2);
figure;
imshow(real(img22),[min(abs(img22(:))) max(abs(img22(:)))]);

figure;
imshow(H11,[min(abs(H11(:))) max(abs(H11(:)))]);
colormap jet;
colorbar;

figure;
imshow(H12,[min(abs(H12(:))) max(abs(H12(:)))]);
colormap jet;
colorbar;

figure;
imshow(H21,[min(abs(H21(:))) max(abs(H21(:)))]);
colormap jet;
colorbar;

figure;
imshow(H22,[min(abs(H22(:))) max(abs(H22(:)))]);
colormap jet;
colorbar;