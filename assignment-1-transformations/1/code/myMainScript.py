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

def part_f(barbara):
    ##################### Part(f) ########################
    imageWithColorbar(barbara)
    M, N = barbara.shape
    new = np.zeros((M,N))
    Cx, Cy = math.ceil(M/2), math.ceil(N/2)
    cos30 = math.cos(math.pi/6)
    sin30 = math.sin(math.pi/6)
    for i in range(M):
        for j in range(N):
            x = ((i-Cx)*cos30 - (j-Cy)*sin30 + Cx)
            y = ((j-Cy)*cos30 + (i-Cx)*sin30 + Cy)
            x = math.ceil(x)
            y = math.ceil(y)
            if((x>0) and (x<M) and (y>0) and (y<N)):
                new[i,j] = barbara[x,y]

    bilinbarbara = myBilinearInterpolation(new, M, N)
    imageWithColorbar(bilinbarbara)

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
    part_f(barbara)

    

