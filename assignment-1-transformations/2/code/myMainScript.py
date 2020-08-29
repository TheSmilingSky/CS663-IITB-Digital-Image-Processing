import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def imageWithColorbar(image,cmap='gray'):
    plt.imshow(image, cmap, interpolation = None)
    plt.colorbar()
    plt.show()

def plotImages(images, cmap=None):
    _, axes = plt.subplots(1,len(images))
    for i in range(len(images)):
        axes[i].imshow(images[i], cmap)
    plt.show()

def plotHist(hists):
    _, axes = plt.subplots(1,len(hists))
    for i in range(len(hists)):
        axes[i].bar(list(range(256)),hists[i])
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

def myHM(img, img_ref):
    img_ref = np.clip(img_ref*255,0,255).astype(int)
    H_ref = imhist(img_ref)
    cdf_ref = np.array([sum(H_ref[:i+1]) for i in range(len(H_ref))])
    sk_ref = np.uint8(255*cdf_ref)

    img = np.clip(img*255,0,255).astype(int)
    H = imhist(img)
    cdf = np.array([sum(H[:i+1]) for i in range(len(H))])
    sk = np.uint8(255*cdf)

    num = [i for i in range(256)]
    new_sk = np.interp(sk, sk_ref, num)
    m,n = img.shape
    Y = np.zeros_like(img)
    for i in range(m):
        for j in range(n):
                Y[i,j] = new_sk[img[i,j]]

    H_new = imhist(Y)

    return Y, H_ref, H, H_new 

def part_a(img):
    imageWithColorbar(img)
    myForegroundMask(img, 0.1)

def part_b(imgs):
    for i in range(len(imgs)):
        imageWithColorbar(imgs[i])
        myLinearContrastStretching(imgs[i])

def part_c(imgs, retina, retina_ref):
    for img in imgs:
        if len(img.shape)==2:
            eqImg, H, HE = myHE(img)
            plotImages([img, eqImg], 'gray')
            plotHist([H, HE])
        else:
            R, HR, HER = myHE(img[:,:,0])
            G, HG, HEG = myHE(img[:,:,1])
            B, HB, HEB = myHE(img[:,:,2])
            
            eqImg = np.dstack((R,G,B))
            plotImages([img, eqImg])
            plotHist([HR, HER])
            plotHist([HG, HEG])
            plotHist([HB, HEB])

def part_d(img, img_ref):
    Rm, HRR, HR, HnR = myHM(img[:,:,0], img_ref[:,:,0])
    Gm, HRG, HG, HnG = myHM(img[:,:,1], img_ref[:,:,1])
    Bm, HRB, HB, HnB = myHM(img[:,:,2], img_ref[:,:,2])
    
    mImg = np.dstack((Rm,Gm,Bm))

    R, HR, HER = myHE(img[:,:,0])
    G, HG, HEG = myHE(img[:,:,1])
    B, HB, HEB = myHE(img[:,:,2])
    
    eqImg = np.dstack((R,G,B))

    plotImages([img,mImg,eqImg])
    # plotHist(HRR, HR, HnR)


if __name__=='__main__':

    barbara = mpimg.imread('../data/barbara.png')
    TEM = mpimg.imread('../data/TEM.png')
    canyon = mpimg.imread('../data/canyon.png')
    retina = mpimg.imread('../data/retina.png')
    church = mpimg.imread('../data/church.png')
    chestXray = mpimg.imread('../data/chestXray.png')
    statue = mpimg.imread('../data/statue.png')
    retina_ref = mpimg.imread('../data/retinaRef.png')
    imgs = [barbara,TEM,canyon,church,chestXray]
    
    part_a(statue)
    part_b(imgs)
    part_c(imgs)
    part_d(retina, retina_ref)
    