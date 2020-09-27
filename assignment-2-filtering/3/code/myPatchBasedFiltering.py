import numpy as np
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

def corrupt_img(img):
    sigma = 0.05 * (np.max(img)-np.min(img))
    noise = np.random.normal(0,sigma,[np.size(img,0),np.size(img,1)])
    corrupt = img + noise
    return corrupt

def padding(img, window, patch):
    pad_img = np.pad(img, np.floor(window/2)+np.floor(patch/2), 'edge')

if __name__=='__main__':
    grass = mpimg.imread('../data/grass.png')
    honey = mpimg.imread('../data/honeyCombReal.png')
    corr_grass = corrupt_img(grass)
    corr_honey = corrupt_img(honey)
    plotImages([grass, corr_grass])
    plotImages([honey, corr_honey])
