function [D, img_mask, bg_mask, newImg] = mySpatiallyVaryingKernel(img, d_thres, xlabel, ylabel, k)
%MYSPATIALLYVARYINGKERNEL Summary of this function goes here
%   Detailed explanation goes here
chan = size(img, 3);

newImg = double(img);
img_g = rgb2gray(img);
% img_mask = zeros(size(img_g));
bg_mask = zeros(size(img_g));

%% Remaining function
[img_mask, img] = myBinaryMask(img, xlabel, ylabel, k);
img_mask = im2double(img_mask);
img_mask = rgb2gray(img_mask);
img = im2double(img);
D = bwdist(img_mask); % Distance
%     imshow(img_mask)
%     title('Segmented Image')
%     pause(1)

H = fspecial('disk', d_thres); % Disc mask
% blurred = imfilter(img, H, 'replicate'); % Convolves
blurred = img;
%     imshow(blurred)
%     title('Blurred Image')
%     pause(1)

wait = waitbar(0, "Spatially Varying Filter in progress");
for i=1:1:chan
%     blurred(:,:,i) = conv2(img(:,:,i), H, 'same');
    newImg(:,:,i) = zeros(size(img_mask));
    newImg(:,:,i) = img_mask.* img(:,:,i);
    bg_mask = ones(size(img_mask)) - img_mask; % background mask
    newImg(:,:,i) = newImg(:,:,i) + bg_mask.* blurred(:,:,i);
    for D_row=1:size(D, 1)
        for D_col = 1:size(D, 2)
            d_local = floor(double(D(D_row, D_col)));
            if d_local <= d_thres && d_local > 0
                H_local = fspecial('disk', d_local);
%                 class(H_local)
%                 size(H_local)
%                 pause(2)
                rows = round(mod( floor(D_row-d_local)-1:ceil(D_row+d_local)-1 ,size(D,1) )+1);
                cols = round(mod( floor(D_col-d_local)-1:ceil(D_col+d_local)-1 ,size(D,2) )+1);
                A = img(rows, cols, i);
%                 class(A)
%                 size(A)
                newImg(D_row, D_col,i)= sum(sum(H_local.*A));
            end
            if d_local > d_thres
                H_local = fspecial('disk', d_thres);
%                 class(H_local)
%                 size(H_local)
%                 pause(2)
                rows = round(mod( floor(D_row-d_thres)-1:ceil(D_row+d_thres)-1 ,size(D,1) )+1);
                cols = round(mod( floor(D_col-d_thres)-1:ceil(D_col+d_thres)-1 ,size(D,2) )+1);
                A = img(rows, cols, i);
%                 class(A)
%                 size(A)
                newImg(D_row, D_col,i)= sum(sum(H_local.*A));
            end
        end
        waitbar(double(i-1)/double(chan) + (double(D_row))/(3 * double(size(D, 1)))); 
    end
end
close(wait);
end