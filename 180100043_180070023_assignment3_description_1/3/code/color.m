function color(img)
    myNumOfColors = 200;
    myColorScale = [[0:1/(myNumOfColors-1):1]', [0:1/(myNumOfColors-1):1]', [0:1/(myNumOfColors-1):1]'];
    imagesc(img);
    colormap (myColorScale);
    colormap jet;
    daspect ([1 1 1]);
    colorbar
end