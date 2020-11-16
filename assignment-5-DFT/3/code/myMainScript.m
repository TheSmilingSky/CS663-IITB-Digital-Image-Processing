%% MyMainScript

tic;
%% Log of Fourier Plot
imgdata = load('../data/image_low_frequency_noise.mat');
img = mat2gray(imgdata.Z);
[m,n] = size(img);
figure;
imshow(img);
img = padarray(img,[m/2,n/2],0);
fourier = fftshift(fft2(img));
fourier_log_mag = log(1+abs(fourier));
figure;
imshow(mat2gray(fourier_log_mag));
colormap jet;
colorbar;
impixelinfo;
toc;

%% finding interfering frequencies
R  = 10;
eta = 0.8;
rj = zeros(0,2);
Max_val = max(fourier_log_mag(:));
H = zeros(2*m,2*n);
for i = 1:2*m
    for j = 1:2*n
        if ((i-m)^2 + (j-n)^2>=R^2) && (fourier_log_mag(i,j)>=eta*Max_val)
            rj(end+1,:) = [i,j];
        end
    end
end

%% Ideal notch filter
R  = 8;
num = size(rj,1);
H = ones(2*m,2*n);
for k=1:num
    H1 = ones(2*m,2*n);
    for i=1:2*m
        for j=1:2*n
            if ((i-rj(k,1))^2 + (j-rj(k,2))^2 <= R^2)
                H1(i,j) = 0;
            end
        end
    end
    H = H.*H1;
end
filtered_img_fourier = H.*fourier;
filtered_img_fourier_log_mag = log(1+abs(filtered_img_fourier));
figure;
imshow(mat2gray(filtered_img_fourier_log_mag));
colormap jet;
colorbar;

filtered_img = ifft2(ifftshift(filtered_img_fourier));
figure;
imshow(real(filtered_img(m/2+1:3*m/2,n/2+1:3*n/2)));
