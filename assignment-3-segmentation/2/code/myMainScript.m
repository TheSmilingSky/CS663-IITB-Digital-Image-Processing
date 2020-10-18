tic;

img = imread("flower.jpg");
img2 = imread("bird.jpg");
img3 = imread("baboonColor.png");

h_space = 200;
h_color = 1;
num_neighbors = 200;
max_iter = 20;

out_img = myMeanShiftSegmentation(img, h_space, h_color, num_neighbors, max_iter);
% % image = Image("../images/flower.png");
% % add(R, image);
fig1 = figure(1);
subplot(1,2,1);
color(img);
title("Original Image");
subplot(1,2,2);
color(out_img);
title("Segmented Image");
saveas(fig1,'baboons_segmented.png')
% caption = Paragraph("Ficg 1: Mean Shift Segmentation applied to flower.png");