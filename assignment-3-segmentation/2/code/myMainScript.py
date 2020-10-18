import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.neighbors import NearestNeighbors

def rescale(img, factor):
    img = Image.fromarray(img)
    w, h = img.size
    img_resize = img.resize((int(w/factor),int(h/factor)))
    return img_resize

def myMeanShiftSegmentation(image, h_spatial, h_intensity, num_neighbours, max_iter):

    img_resized = rescale(image, 20)
    segmented = image
    rows, columns = np.size(image,0), np.size(image,1)
    attr_size = rows*columns
    attr = np.zeros((attr_size,5))
    for i in range(rows):
        for j in range(columns):
            attr[(i)*columns+j,:] = i/h_spatial, j/h_spatial, image[i,j,0]/h_intensity, image[i,j,1]/h_intensity, image[i,j,2]/h_intensity

    num_iter = 0
    while num_iter < max_iter:
        print(num_iter/max_iter*100)
        nbrs = NearestNeighbors(n_neighbors=num_neighbours).fit(attr)
        print('a')
        distances, indices = nbrs.kneighbors(attr)
        print('b')
        temp_attr = attr
        print('c')
        for i in range(attr_size):
            print(i/attr_size*100)
            weights = np.exp(-(np.square(distances[i,:]))/2)
            sum_weights = np.sum(weights, axis=0)
            weights = np.transpose(weights).reshape(-1,1)
            print(weights.shape)
            weight_arr = np.concatenate((weights, weights, weights),axis=1)
            print(weight_arr.shape)
            attr[i, 2:5] = np.sum(np.multiply(weight_arr,temp_attr[indices[i,:],2:5]))/sum_weights

        num_iter += 1

    for i in range(rows):
        for j in range(columns):
            segmented[i,j,0] = attr[i*columns+j,2]
            segmented[i,j,1] = attr[i*columns+j,3]
            segmented[i,j,2] = attr[i*columns+j,4]

    segmented = rescale(segmented, 0.5)

    return segmented

if __name__=='__main__':
    bird = mpimg.imread('../data/bird.jpg')
    image = myMeanShiftSegmentation(bird, 100, 1, 200, 15)
    plt.imshow(image)
    plt.show()