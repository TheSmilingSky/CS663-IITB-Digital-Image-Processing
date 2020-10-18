tic;

img = imread("../data/flower.jpg");
img2 = imread("../data/bird.jpg");
img3 = imread("../data/baboonColor.png");

h_space = 200;
h_color = 1;
num_neighbors = 200;
max_iter = 20;

out_img = myMeanShiftSegmentation(img, h_space, h_color, num_neighbors, max_iter);
imwrite(out_img,'flower1.png');
% % image = Image("../images/flower.png");
% % add(R, image);
fig1 = figure(1);
subplot(1,2,1);
color(img);
title("Original Image");
subplot(1,2,2);
color(out_img);
title("Segmented Image");
saveas(fig1,'flowersegment.png')
% caption = Paragraph("Ficg 1: Mean Shift Segmentation applied to flower.png");
out_img = myMeanShiftSegmentation(img2, h_space, h_color, num_neighbors, max_iter);
imwrite(out_img,'birds1.png');
fig2 = figure(1);
subplot(1,2,1);
color(img2);
title("Original Image");
subplot(1,2,2);
color(out_img);
title("Segmented Image");
saveas(fig2,'birdsegment.png')
out_img = myMeanShiftSegmentation(img3, h_space, h_color, num_neighbors, max_iter);
imwrite(out_img,'baboon1.png');
fig3 = figure(1);
subplot(1,2,1);
color(img3);
title("Original Image");
subplot(1,2,2);
color(out_img);
title("Segmented Image");
saveas(fig3,'baboonssegment.png')