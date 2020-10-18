function [img_mask,out_img] = myBinaryMask(img, xlabel, ylabel, k)
    mask = img;
    wait = waitbar(0, "Clustering");
    clustered = imsegkmeans(img,k);
    rows = size(mask,1);
    columns = size(mask,2);
    kdes = clustered(xlabel,ylabel);
    for i=1:rows
        for j=1:columns
            if clustered(i,j)== kdes
                mask(i,j,:) = 1;
            else 
                mask(i,j,:) = 0;
            end
        end
    end
    img_mask = 255*mask;
    out_img = mask.*img;
    figure;
    imshow(255*mask);
    figure;
    imshow(mask.*img);
    figure;
    imshow((1-mask).*img);
    close(wait);
end

