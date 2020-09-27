import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def plotImages(images, cmap='gray'):
    _, axes = plt.subplots(1,len(images))
    for i in range(len(images)):
        axes[i].imshow(images[i], cmap)
    plt.show()

def imhist(img):
    m,n = img.shape
    # img = np.clip(img*255,0,255).astype(int)
    H = [0.0]*256
    for i in range(m):
        for j in range(n):
            H[img[i,j]] += 1
    return np.array(H)/(m*n)

def median(hist):
    nz_hist = []
    for i in range(len(hist)):
        if hist[i]!=0:
            nz_hist.append(i)

    return nz_hist[round((len(nz_hist)+1)/2)]

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

def bi_histogram(img):
    img = np.clip(img*255,0,255).astype(int)
    H = imhist(img)
    med = median(H)

    H1, H2 = H[:med], H[med:]
    cdf1, cdf2 = np.array([sum(H1[:i+1]) for i in range(len(H1))]), np.array([sum(H2[:i+1]) for i in range(len(H2))])
    sk1, sk2 = np.uint8(255*cdf1), np.uint8(255*cdf2)
    m,n = img.shape
    Y = np.zeros_like(img)
    for i in range(m):
        for j in range(n):
            if (img[i,j] < med):
                Y[i,j] = sk1[img[i,j]]
            else:
                Y[i,j] = sk2[img[i,j]-med] + med
    return Y


if __name__ == '__main__':
    img = mpimg.imread('third.jpg')
    bi_img = bi_histogram(img[:,:,0])
    eqImg, _, _ = myHE(img[:,:,0])
    plotImages([img, bi_img, eqImg])