import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import math
from math import floor

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

def myPadding(img, window, patch):
    pad_img = np.pad(img, floor(window/2)+ floor(patch/2), 'edge')
    return pad_img

def myPatchBasedFiltering(img, var):
	result = np.zeros(img.shape)
	window = 25
	patchSize = 9
	pad = floor(window/2)+floor(patchSize/2)
	paddedImage = myPadding(img, window, patchSize)
	gaussianMask = np.zeros((patchSize,patchSize))
	sigma = 1
	for x in range(-floor(patchSize/2),floor(patchSize/2)):
		for y in range(-floor(patchSize/2),floor(patchSize/2)):
			gaussianMask[x,y] = np.exp(-(x**2-y**2)/2*sigma**2)

	for i in range(pad, img.shape[0]+pad):
		i1 = max(i - floor(window/2), 0)
		i2 = min(i + floor(window/2), paddedImage.shape[0])

		for j in range(pad, img.shape[1]+pad):
			j1 = max(j - floor(window/2), 0)
			j2 = min(j + floor(window/2), paddedImage.shape[0])

			borders = [i1,i2,j1,j2]
			a1 = max(int(0.5*(borders[0]+borders[1])) - floor(patchSize/2),0)
			a2 = min(int(0.5*(borders[0]+borders[1])) + floor(patchSize/2), paddedImage.shape[0])
			b1 = max(int(0.5*(borders[2]+borders[3])) - floor(patchSize/2),0)
			b2 = min(int(0.5*(borders[2]+borders[3])) + floor(patchSize/2), paddedImage.shape[1])
			patch = np.multiply(paddedImage[a1:a2+1,b1:b2+1],gaussianMask)
			W = np.zeros((i2-i1+1,j2-j1+1))

			for m in range(i1,i2+1):
				pi1 = max(m - floor(patchSize/2),0)
				pi2 = min(m + floor(patchSize/2),paddedImage.shape[0])

				for n in range(j1,j2+1):
					pj1 = max(n - floor(patchSize/2),0)
					pj2 = min(n + floor(patchSize/2),paddedImage.shape[1])

					patch1 = np.multiply(paddedImage[pi1:pi2+1,pj1:pj2+1],gaussianMask)
					W[m-i1,n-j1] = np.exp(-sum(sum(np.square(patch - patch1)))/sigma**2)
			# print(np.multiply(W,borders[i1:i2+1,j1:j2+1]))

			result[i-pad,j-pad] = sum(sum(np.multiply(W,paddedImage[i1:i2+1,j1:j2+1])))/sum(sum(W))
		print("done", i, img.shape[0])
	return result



if __name__=='__main__':
    grass = mpimg.imread('../data/grass.png')
    honey = mpimg.imread('../data/honeyCombReal.png')
    corr_grass = corrupt_img(grass)
    corr_honey = corrupt_img(honey)
    filter_grass = myPatchBasedFiltering(corr_grass, 0.15)
    filter_honey = myPatchBasedFiltering(corr_honey, 0.15)
    plotImages([grass, corr_grass, filter_grass])
    plotImages([honey, corr_honey, filter_honey])
