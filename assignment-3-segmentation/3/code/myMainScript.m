%% MyMainScript

tic;
flower = imread('../images/flower1.png');
bird = imread('../images/birds1.png');
myBinaryMask(flower,120,200,2);
myBinaryMask(bird,330,330,2);
toc;
