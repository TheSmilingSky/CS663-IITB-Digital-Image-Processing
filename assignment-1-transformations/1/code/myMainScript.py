from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

def myShrinkImageByFactorD(img, k):
    return img[::k,::k]

def myBilinearInterpolation(img):
    M,N = img.shape
    P,Q = 3*M-2, 2*N-1
    new = np.zeros((P,Q))
    new[::3,::2] = img
    for i in range(1,Q,2):
        new[:,i] = (new[:,i-1]+new[:,i+1])/2
    for j in range(1,P,3):
        new[j,:] = (2*new[j-1,:]+new[j+2,:])/3
        new[j+1,:] = (new[j-1,:]+2*new[j+2,:])/3
    return new

def imageWithColorbar(image,cmap='gray'):
    image = plt.imshow(image, cmap, interpolation = None)
    plt.colorbar()
    plt.show()

def part_a(circles, barbara):
    ##################### Part(a) ########################
    imageWithColorbar(circles)
    circles_shrink2 = myShrinkImageByFactorD(np.array(circles),2)
    circles_shrink3 = myShrinkImageByFactorD(np.array(circles),3)
    imageWithColorbar(circles_shrink2)
    imageWithColorbar(circles_shrink3)

def part_b(circles, barbara):
    ##################### Part(b) ########################
    imageWithColorbar(barbara)
    enlarged_barbara = myBilinearInterpolation(barbara)
    imageWithColorbar(enlarged_barbara)    

if __name__=='__main__':
        
    circles = mpimg.imread('../data/circles_concentric.png')
    barbara = mpimg.imread('../data/barbaraSmall.png')

    part_a(circles, barbara)
    part_b(circles, barbara)