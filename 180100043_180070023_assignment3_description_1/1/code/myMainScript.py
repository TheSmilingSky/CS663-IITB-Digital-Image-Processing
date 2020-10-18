import numpy as np
import math
import mat73
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def imageWithColorbar(image,cmap='gray'):
    plt.imshow(image, cmap, interpolation = None)
    plt.colorbar()
    plt.show()

def plotImages(images, cmap='gray'):
    a, axes = plt.subplots(1,len(images))
    for i in range(len(images)):
        axes[i].imshow(images[i], cmap)
    plt.show()

def gaussianMask(size=3,sigma=1):
    center=(int)(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
       for j in range(size):
          diff=np.sqrt((i-center)**2+(j-center)**2)
          kernel[i,j]=np.exp(-(diff**2)/(2*sigma**2))
    return kernel/np.sum(kernel)

def rescaleImage(img):
	maxI = np.amax(img)
	minI = np.amin(img)
	return (img-minI)/(maxI-minI)

def myHarrisCornerDetection(img, k, sigma1, sigma2):
	mask1 = gaussianMask(3, sigma1)
	dxG, dyG = np.gradient(mask1)

	#Image Derivatives
	dX = convolve2d(img, dxG, 'same')
	dY = convolve2d(img, dyG, 'same')
	plotImages([dX, dY])

	mask2 = gaussianMask(9, sigma2)

	Ixx = convolve2d(dX**2, mask2, 'same')
	Iyy = convolve2d(dY**2, mask2, 'same')
	Ixy = convolve2d(dX*dY, mask2, 'same')
	vals = np.zeros((img.shape[0],img.shape[1],2))
	val1 = np.zeros(img.shape)
	val2 = np.zeros(img.shape)
	C = np.zeros(img.shape)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			A = np.array([[Ixx[i,j], Ixy[i,j]], [Ixy[i,j], Iyy[i,j]]])
			eigenVals, _ = np.linalg.eig(A)
			vals[i,j,0], vals[i,j,1] = eigenVals[0], eigenVals[1]
			C[i,j] = vals[i,j,0]*vals[i,j,1] - k*(vals[i,j,0]+vals[i,j,1])**2
			# print(vals[i,j])
	imageWithColorbar(vals[:,:,0])
	imageWithColorbar(vals[:,:,1])
	imageWithColorbar(C)


if __name__=='__main__':
    boat_dict = mat73.loadmat('../data/boat.mat')
    boat = boat_dict['imageOrig']
    boat = rescaleImage(boat)
    imageWithColorbar(boat)
    myHarrisCornerDetection(boat, 0.01,0.5,1.5)