%% MyMainScript

tic;
%% Your code here

bird = imread('../images/birds1.png');
scaling = 0.4;
bird = imresize(bird, scaling);
flower = imread('../images/flower1.png');
scaling2 = 0.7;
flower = imresize(flower, scaling2);

tic;
%% For bird ->
d_thres = 40;
% r_min = round(75*scaling);
% r_max = round(650*scaling);
% c_min = round(230*scaling);
% c_max = round(770*scaling);
% threshold = 180;
xlabel = round(330*scaling);
ylabel = round(330*scaling);
k = 2;
[D, img_mask, bg_mask, img] = mySpatiallyVaryingKernel(bird, d_thres, xlabel, ylabel, k);
%disp(img);
img_mask = double(img_mask)/255;
foreground = double(img);
background = double(img);
chan = size(img, 3);
for i=1:chan
   foreground(:,:,i) = immultiply(img_mask, double(bird(:,:,i)));
   background(:,:,i) = immultiply(double(bg_mask), double(bird(:,:,i))/255);
end

figure(1);
subplot(1,3,1);
greyscale(img_mask);
title('a) Mask M');
subplot(1,3,2);
color(foreground);
title('b) Foreground');
subplot(1,3,3);
color(background);
title('c) Background');

% imwrite(img_mask, '../images/bird_mask.png');
% imwrite(foreground, '../images/bird_foreground.png');
% imwrite(background, '../images/bird_background.png');
% pause(2);
% close;

figure(2);
contour(flipud(D), 'ShowText', 'on');
title("Variation of Disk Radius");
% saveas(gcf, '../images/bird_contour.png');
% pause(2);

figure(3);
% Need to display these...
subplot(2,3,1);
greyscale(fspecial('disk', round(0.2*d_thres)));
title("Kernel at 0.2 d_{thresh}");
% pause(2);
subplot(2,3,2);
greyscale(fspecial('disk', round(0.4*d_thres)));
title("Kernel at 0.4 d_{thresh}");
% pause(2);
subplot(2,3,3);
greyscale(fspecial('disk', round(0.6*d_thres)));
title("Kernel at 0.6 d_{thresh}");
% pause(2);
subplot(2,3,4);
greyscale(fspecial('disk', round(0.8*d_thres)));
title("Kernel at 0.8 d_{thresh}");
% pause(2);
subplot(2,3,6);
greyscale(fspecial('disk', round(d_thres)));
title("Kernel at d_{thresh}");
% pause(2);

figure(4);
subplot(1,2,1);
color(bird);
title("Original Image");
subplot(1,2,2);
color(img);
title('Spatially varying blurred image');
% imwrite(img, '../images/bird_blurred.png');
% pause(2);
toc;

tic;
%% For flower ->
d_thres = 20;
% r_min = round(65*scaling2);
% r_max = round(215*scaling2);
% c_min = round(150*scaling2);
% c_max = round(280*scaling2);
% threshold = 100;
xlabel = round(120*scaling2);
ylabel = round(200*scaling2);
k = 2;
[D, img_mask, bg_mask, img] = mySpatiallyVaryingKernel(flower, d_thres, xlabel, ylabel, k);
img_mask = double(img_mask)/255;
foreground = double(img);
background = double(img);
chan = size(img, 3);
for i=1:chan
   foreground(:,:,i) = immultiply(img_mask, double(flower(:,:,i)));
   background(:,:,i) = immultiply(double(bg_mask), double(flower(:,:,i))/255);
end
figure(5);
subplot(1,3,1);
greyscale(img_mask);
title('a) Mask M');
subplot(1,3,2);
color(foreground);
title('b) Foreground');
subplot(1,3,3);
color(background);
title('c) Background');
% imwrite(img_mask, '../images/flower_mask.png');
% imwrite(foreground, '../images/flower_foreground.png');
% imwrite(background, '../images/flower_background.png');
% pause(2);
% close;

figure(6);
contour(flipud(D), 'ShowText', 'on');
title("Variation of Disk Radius");
% saveas(gcf, '../images/flower_contour.png');
% pause(2);

figure(7)
subplot(2,3,1);
greyscale(fspecial('disk', round(0.2*d_thres)));
title("Kernel at 0.2 d_{thresh}");
% pause(2);
subplot(2,3,2);
greyscale(fspecial('disk', round(0.4*d_thres)));
title("Kernel at 0.4 d_{thresh}");
% pause(2);
subplot(2,3,3);
greyscale(fspecial('disk', round(0.6*d_thres)));
title("Kernel at 0.6 d_{thresh}");
% pause(2);
subplot(2,3,4);
greyscale(fspecial('disk', round(0.8*d_thres)));
title("Kernel at 0.8 d_{thresh}");
% pause(2);
subplot(2,3,6);
greyscale(fspecial('disk', round(d_thres)));
title("Kernel at d_{thresh}");
% pause(2);

figure(8);
subplot(1,2,1);
color(flower);
title("Original Image");
subplot(1,2,2);
color(img);
title('Spatially varying blurred image');
% imwrite(img, '../images/flower_blurred.jpg');
% pause(2);
toc;