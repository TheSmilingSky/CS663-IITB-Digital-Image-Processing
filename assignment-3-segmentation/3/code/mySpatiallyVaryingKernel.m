function [newImg] = mySpatiallyVaryingKernel(img, d_thres, img_mask)
%MYSPATIALLYVARYINGKERNEL Summary of this function goes here
%   Detailed explanation goes here
% chan = size(img, 3);

% newImg = double(img);
% img_g = rgb2gray(img);
% img_mask = zeros(size(img_g));
% bg_mask = zeros(size(img_g));

%% Remaining function
% img_mask = im2double(img_mask);
% img_mask_2 = rgb2gray(img_mask);
% img = im2double(img);
newImg = img_mask.* img;
bg_mask = 1 - img_mask; % background mask
newImg = newImg+ bg_mask.* img;
D = bwdist(rgb2gray(img_mask)); % Distance
%     imshow(img_mask)
%     title('Segmented Image')
%     pause(1)

% H = fspecial('disk', d_thres); % Disc mask
% blurred = imfilter(img, H, 'replicate'); % Convolves,
%     imshow(blurred)
%     title('Blurred Image')
%     pause(1)

wait = waitbar(0, "Spatially Varying Filter in progress");
%     blurred(:,:,i) = conv2(img(:,:,i), H, 'same');
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
            A = double(img(rows, cols, :));
%                 class(A)
%                 size(A)
            newImg(D_row, D_col,:)= sum(sum(H_local.*A));
        end
        if d_local > d_thres
            H_local = fspecial('disk', d_thres);
%                 class(H_local)
%                 size(H_local)
%                 pause(2)
            rows = round(mod( floor(D_row-d_thres)-1:ceil(D_row+d_thres)-1 ,size(D,1) )+1);
            cols = round(mod( floor(D_col-d_thres)-1:ceil(D_col+d_thres)-1 ,size(D,2) )+1);
            A = double(img(rows, cols, :));
%                 class(A)
%                 size(A)
            newImg(D_row, D_col,:)= sum(sum(H_local.*A));
        end
    end
    waitbar((double(D_row))/(3 * double(size(D, 1)))); 
end
close(wait);
end