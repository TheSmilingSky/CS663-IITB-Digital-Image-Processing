import numpy as np
import math
import mat73
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def imageWithColorbar(image,cmap='gray'):
    plt.imshow(image, cmap, interpolation = None)
    plt.colorbar()
    plt.show()

def plotImages(images, cmap='gray'):
    _, axes = plt.subplots(1,len(images))
    for i in range(len(images)):
        axes[i].imshow(images[i], cmap)
    plt.show()

def get_rmsd(img1, img2):
    assert img1.shape==img2.shape, "shape mismatch"
    print(np.size(img1))
    rmsd = math.sqrt(np.sum(np.square(img1-img2))/(np.size(img1,0)*np.size(img1,1)))
    return rmsd

def corrupt_img(img):
    sigma = 0.05 * (np.max(img)-np.min(img))
    noise = np.random.normal(0,sigma,[np.size(img,0),np.size(img,1)])
    corrupt = img + noise
    return corrupt

def bilateralFilter(img, window, sigma_s, sigma_i):
    filtered_img = np.zeros((np.size(img,0),np.size(img,1)))
    x_m = [i for i in range(int(-1*np.floor(window/2)), int(np.floor(window/2)+1))]
    x,y = np.meshgrid(x_m, x_m)
    G_s = np.exp(-(np.square(x)+np.square(y))/(2*sigma_s*sigma_s))

    for i in range(np.size(img,0)):
        i_left = max(i - int(np.floor(window/2)), 0)
        i_right = min(i + int(np.floor(window/2))+1, np.size(img,0))
        
        for j in range(np.size(img,1)):
            j_left = max(j - int(np.floor(window/2)), 0)
            j_right = min(j + int(np.floor(window/2))+1, np.size(img,1))
            
            p_intensity = img[i,j]
            intensity_window = img[i_left:i_right,j_left:j_right]
            
            G_i = np.exp(-np.square(intensity_window - p_intensity)/(2*sigma_i*sigma_i))
            G_s_new = G_s[i_left-i+int(np.floor(window/2)):i_right-i+int(np.floor(window/2)),j_left-j+int(np.floor(window/2)):j_right-j+int(np.floor(window/2))]
            assert G_s_new.shape == G_i.shape, "shape mismatch"
            weights = np.multiply(G_s_new, G_i)
            filtered_img[i,j] = np.sum(np.multiply(intensity_window,weights))/(np.sum(weights))

    return filtered_img

def padding(img, window, patch):
    pad_img = np.pad(img, np.floor(window/2)+np.floor(patch/2), 'edge')

if __name__=='__main__':
    data_dict = mat73.loadmat('../data/barbara.mat')
    barbara = data_dict['imageOrig']
    grass = mpimg.imread('../data/grass.png')
    honey = mpimg.imread('../data/honeyCombReal.png')
    corr_barbara = corrupt_img(barbara)
    corr_grass = corrupt_img(grass)
    corr_honey = corrupt_img(honey)
    filter_barbara = bilateralFilter(corr_barbara, 50, 10, 10)
    filter_grass = bilateralFilter(corr_grass, 9, 0.81, 0.18)
    filter_honey = bilateralFilter(corr_honey, 9, 0.88, 0.16)
    plotImages([barbara, corr_barbara, filter_barbara])
    plotImages([grass, corr_grass, filter_grass])
    plotImages([honey, corr_honey, filter_honey])
    print("RMSD1 : ", get_rmsd(barbara,filter_barbara))
    print("RMSD2 : ", get_rmsd(grass,filter_grass))
    print("RMSD3 : ", get_rmsd(honey,filter_honey))
