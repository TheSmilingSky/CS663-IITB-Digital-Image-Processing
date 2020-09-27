import numpy as np
import math
import mat73
from scipy.ndimage import gaussian_filter
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

def linear_contrast_stretch(img):
    img = img/(np.max(img)-np.min(img))
    return img

def myUnsharpMasking(img, sigma, scale):
    unsharp_image = gaussian_filter(img, sigma)
    unsharp_mask = img - unsharp_image

    sharp_image = img + scale*unsharp_mask
    return sharp_image

if __name__=='__main__':
    moon_dict = mat73.loadmat('../data/superMoonCrop.mat')
    moon = moon_dict['imageOrig']
    lion_dict = mat73.loadmat('../data/lionCrop.mat')
    lion = lion_dict['imageOrig']
    moon_sharp = myUnsharpMasking(moon, 6, 0.5)
    lion_sharp = myUnsharpMasking(lion, 6, 0.5)
    plotImages([linear_contrast_stretch(moon), linear_contrast_stretch(moon_sharp)])
    plotImages([linear_contrast_stretch(lion), linear_contrast_stretch(lion_sharp)])