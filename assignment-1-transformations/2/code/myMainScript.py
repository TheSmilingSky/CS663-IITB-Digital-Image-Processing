import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def imageWithColorbar(image,cmap='gray'):
    plt.imshow(image, cmap, interpolation = None)
    plt.colorbar()
    plt.show()

def plotImages(im1, im2, cmap=None):
    _, axes = plt.subplots(1,2)
    axes[0].imshow(im1, cmap)
    axes[1].imshow(im2, cmap)
    plt.show()

def plotHist(h1, h2):
    _, axes = plt.subplots(1,2)
    axes[0].bar(list(range(256)),h1)
    axes[1].bar(list(range(256)),h2)
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
    if len(img.shape)==2:
        plt.imshow(np.clip(img,0,255).astype(int), cmap='gray')
    else:
        plt.imshow(np.clip(img,0,255).astype(int))
    plt.colorbar()
    plt.show()

def imhist(img):
    m,n = img.shape
    # img = np.clip(img*255,0,255).astype(int)
    H = [0.0]*256
    for i in range(m):
        for j in range(n):
            H[img[i,j]] += 1
    return np.array(H)/(m*n)
 
def myHE(img):
    img = np.clip(img*255,0,255).astype(int)
    H = imhist(img)
    cdf = np.array([sum(H[:i+1]) for i in range(len(H))])
    sk = np.uint8(255*cdf)
    m,n = img.shape
    Y = np.zeros_like(img)
    for i in range(m):
        for j in range(n):
            Y[i,j] = sk[img[i,j]]
    HE = imhist(Y)
    return Y, H, HE
    
def part_a(img):
    imageWithColorbar(img)
    myForegroundMask(img, 0.1)

def part_b(imgs):
    for i in range(len(imgs)):
        imageWithColorbar(imgs[i])
        myLinearContrastStretching(imgs[i])

def part_c(imgs):
    for img in imgs:
        if len(img.shape)==2:
            eqImg, H, HE = myHE(img)
            plotImages(img, eqImg, 'gray')
            plotHist(H, HE)
        else:
            R, HR, HER = myHE(img[:,:,0])
            G, HG, HEG = myHE(img[:,:,1])
            B, HB, HEB = myHE(img[:,:,2])
            eqImg = np.dstack((R,G,B))
            plotImages(img, eqImg)
            plotHist(HR, HER)
            plotHist(HG, HEG)
            plotHist(HB, HEB)

if __name__=='__main__':

    barbara = mpimg.imread('../data/barbara.png')
    TEM = mpimg.imread('../data/TEM.png')
    canyon = mpimg.imread('../data/canyon.png')
    retina = mpimg.imread('../data/retina.png')
    church = mpimg.imread('../data/church.png')
    chestXray = mpimg.imread('../data/chestXray.png')
    statue = mpimg.imread('../data/statue.png')
    imgs = [barbara,TEM,canyon,church,chestXray]
    
    part_a(statue)
    part_b(imgs)
    part_c(imgs)
