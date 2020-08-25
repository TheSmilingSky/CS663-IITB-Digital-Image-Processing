import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def imageWithColorbar(image,cmap='gray'):
    if len(image.shape)==2:
        plt.imshow(np.clip(image,0,255).astype(int), cmap='gray')
    else:
        plt.imshow(np.clip(image,0,255).astype(int))
    plt.colorbar()
    plt.show()

def myForegroundMask(img, threshold):
    mask = img.copy()
    mask = mask>threshold
    imageWithColorbar(mask)
    masked = np.multiply(img,mask)
    imageWithColorbar(masked)

def myLinearContrastStretching(img):
    L = 255
    minI, maxI = np.ndarray.min(img), np.ndarray.max(img)
    img = (img*255/(maxI-minI))
    img = np.multiply(img/2,img<L/8)+np.multiply(-L/4 + 3*img/2,np.multiply(img>=L/8,img<=7*L/8))+np.multiply(3*L/4+img/2,img>7*L/8) 
    minI, maxI = np.ndarray.min(img), np.ndarray.max(img)
    img = (img*255/(maxI-minI))
    imageWithColorbar(img, cmap=None)

def part_a(img):
    imageWithColorbar(img)
    myForegroundMask(img, 0.1)

def part_b(imgs):
    for i in range(len(imgs)):
        myLinearContrastStretching(imgs[i])

if __name__=='__main__':

    barbara = mpimg.imread('../data/barbara.png')
    TEM = mpimg.imread('../data/TEM.png')
    canyon = mpimg.imread('../data/canyon.png')
    retina = mpimg.imread('../data/retina.png')
    church = mpimg.imread('../data/church.png')
    chestXray = mpimg.imread('../data/chestXray.png')
    statue = mpimg.imread('../data/statue.png')
    
    part_a(statue)

    imgs_part_b = [barbara,TEM,canyon,church,chestXray]
    part_b(imgs_part_b)
