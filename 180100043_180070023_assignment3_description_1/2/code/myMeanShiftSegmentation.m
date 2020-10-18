function [segmented_img] = myMeanShiftSegmentation(img, h_spatial, h_intensity, num_neighbours, max_iter)

    img = im2double(img); 
    segmented_img = img;

    rows = size(img, 1);
    columns = size(img, 2);
    feature_size = rows*columns;
    features = zeros(feature_size, 5);
    for i=1:rows
        for j=1:columns
            features((i-1)*columns+j,:) = [i/h_spatial, j/h_spatial,...
                img(i,j,1)/h_intensity, img(i,j,2)/h_intensity, img(i,j,3)/h_intensity];
        end
    end
    wait = waitbar(0, "Mean Shift Segmentation in progress");
    num_iter = 0;
    while(num_iter < max_iter)
        [nearest_neigh, distances] = knnsearch(features, features, 'k', num_neighbours);
        temp_features = features;
        for i=1:feature_size
            weights = exp(-(distances(i,:).^2)/2);
            sum_weights = sum(weights);
            weights = weights';
            weight_arr = [weights, weights, weights];
            features(i, 3:5) = sum(weight_arr.*temp_features(nearest_neigh(i,:),3:5))/sum_weights;
        end
        num_iter = num_iter + 1;
        waitbar(double(num_iter)/double(max_iter)); 
    end
    
    for i=1:rows
        for j=1:columns
            segmented_img(i,j,1) = features((i-1)*columns+j,3);
            segmented_img(i,j,2) = features((i-1)*columns+j,4);
            segmented_img(i,j,3) = features((i-1)*columns+j,5);
        end
    end
    close(wait);
end