function myBinaryMask(img, xlabel, ylabel, k)
    wait = waitbar(0, "Clustering");
    clustered = imsegkmeans(img,k);
    rows = size(img,1);
    columns = size(img,2);
    mask = img;
    kdes = clustered(xlabel,ylabel);
    for i=1:rows
        for j=1:columns
            if clustered(i,j)==kdes
                mask(i,j,:) = 1;
            else 
                mask(i,j,:) = 0;
            end
        end
    end
    figure;
    imshow(255*mask);
    figure;
    imshow(mask.*img);
    figure;
    imshow((1-mask).*img);
    close(wait);
end