import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

def myShrinkImageByFactorD(img, k):
    return img[::k,::k]

def myBilinearInterpolation(img,P,Q):
    M,N = img.shape
    new = np.zeros((P,Q))
    r,c = math.ceil(P/M),math.ceil(Q/N)
    new[::r,::c] = img
    for i in range(c,Q,c):
        for j in range(0,P,r):
            new[j,i-c+1:i] = np.linspace(new[j,i-c],new[j,i],c+1)[1:-1]
    for i in range(r,P,r):
        for j in range(0,Q,1):
            new[i-r+1:i,j] = np.linspace(new[i-r,j],new[i,j],r+1)[1:-1]
    return new

def myNearestNeighborInterpolation(img, P, Q):
    M, N = img.shape
    new = np.zeros((P, Q))
    
    r,c = math.ceil(P/M), math.ceil(Q/N)
    new[::r, ::c] = img
    for j in range(0,Q,c):
        for i in range(0,P-r,r):
            new[i+1,j] = new[i,j]
            new[i+r-1, j] = new[i+r,j]

    for j in range(0, Q-c, c):
        for i in range(0, P-r,r):
            new[i+1,j+1] = new[i,j]
            new[i+r-1, j+c-1] = new[i+r, j]
    
    for i in range(0,P, r):
        for j in range(0,Q-c,c):
            new[i, j+1] = new[i, j]

    return new

def imageWithColorbar(image,cmap='gray'):
    image = plt.imshow(image, cmap, interpolation = None)
    plt.colorbar()
    plt.show()

def part_a(circles):
    ##################### Part(a) ########################
    imageWithColorbar(circles)
    circles_shrink2 = myShrinkImageByFactorD(np.array(circles),2)
    circles_shrink3 = myShrinkImageByFactorD(np.array(circles),3)
    imageWithColorbar(circles_shrink2)
    imageWithColorbar(circles_shrink3)

def part_b(barbara):
    ##################### Part(b) ########################
    imageWithColorbar(barbara)
    M,N = barbara.shape
    enlarged_barbara = myBilinearInterpolation(barbara,3*M-2,2*N-1)
    imageWithColorbar(enlarged_barbara)    

def part_c(barbara):
    ##################### Part(c) ########################
    imageWithColorbar(barbara)
    M,N = barbara.shape
    enlarged_barbara = myNearestNeighborInterpolation(barbara,3*M-2,2*N-1)
    imageWithColorbar(enlarged_barbara)    


if __name__=='__main__':
        
    circles = mpimg.imread('../data/circles_concentric.png')
    barbara = mpimg.imread('../data/barbaraSmall.png')

    part_a(circles)
    part_b(barbara)
    part_c(barbara)