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
    for i in range(P):
        ii = round(i/r)
        if ii<1:
            ii = 1
        for j in range(Q):
            jj = round(j/c)
            if jj<1:
                jj = 1
            new[i,j] = img[ii,jj]

    return new

def myBicubicInterpolation(img, P, Q):
    M, N = img.shape
    r,c = math.ceil(P/M), math.ceil(Q/N)
    new = np.zeros((P, Q))
    new[::r, ::c] = img
    mat = np.array([[1,0,0,0],[0,0,1,0],[-3,3,-2,-1],[2,-2,1,1]])
    f = img.copy()
    fx = np.zeros((M,N))
    fy = np.zeros((M,N))
    fxy = np.zeros((M,N))

    for i in range(M):
        for j in range(N):
            if j == 0 or j == N-1:
                fx[i,j] = 0
            else:
                fx[i, j] = 0.5*(f[i,j+1] - f[i,j-1])
            
            if i == 0 or i == M-1:
                fy[i,j] = 0
            else:
                fy[i, j] = 0.5*(f[i+1,j] - f[i-1,j])
            
            if i == 0  or i == M-1 or j == 0 or j == N-1:
                fxy[i, j] = 0
            else:
                fxy[i, j] = 0.25*(f[i+1,j+1] - f[i+1,j-1] - f[i-1,j+1] + f[i-1,j-1])
    
    for i in range(P-1):
        for j in range(Q-1):
            i1, j1 = math.floor(i/r), math.floor(j/c)
            i2, j2 = math.ceil(i/r), math.ceil(j/c)
            if i2<M and j2<N:
                F = (np.array([
                    [f[i1,j1],f[i1,j2],fy[i1,j1],fy[i1,j2]],
                    [f[i2,j1],f[i2,j2],fy[i2,j1],fy[i2,j2]],
                    [fx[i1,j1],fx[i1,j2],fxy[i1,j1],fxy[i1,j2]],
                    [fx[i2,j1],fx[i2,j2],fxy[i2,j1],fxy[i2,j2]],
                ]))
            params = np.matmul(mat,np.matmul(F,np.transpose(mat)))
            x_, y_ = i/r - i1, j/c - j1
            X,Y = np.array([1,x_,x_**2,x_**3]), np.transpose([1,y_,y_**2,y_**3])
            val = np.matmul(X,np.matmul(params,Y))
            new[i+1,j+1] = float(val)
    return new

def myImageRotation(img, theta):
    M, N = img.shape
    new = np.zeros((M,N))
    Cx, Cy = math.ceil(M/2), math.ceil(N/2)
    c = math.cos(theta)
    s = math.sin(theta)
    for i in range(M):
        for j in range(N):
            x = ((i-Cx)*c - (j-Cy)*s + Cx)
            y = ((j-Cy)*c + (i-Cx)*s + Cy)
            x = math.ceil(x)
            y = math.ceil(y)
            if((x>0) and (x<M) and (y>0) and (y<N)):
                new[i,j] = img[x,y]

    return myBilinearInterpolation(new, M, N)

def imageWithColorbar(image,cmap='gray'):
    image = plt.imshow(image, cmap, interpolation = None)
    plt.colorbar()
    plt.show()

def part_a(img):
    ##################### Part(a) ########################
    imageWithColorbar(img)
    shrink2 = myShrinkImageByFactorD(np.array(img),2)
    shrink3 = myShrinkImageByFactorD(np.array(img),3)
    imageWithColorbar(shrink2)
    imageWithColorbar(shrink3)

def part_b(img):
    ##################### Part(b) ########################
    imageWithColorbar(img)
    M,N = img.shape
    enlarged = myBilinearInterpolation(img,3*M-2,2*N-1)
    imageWithColorbar(enlarged)    

def part_c(img):
    ##################### Part(c) ########################
    imageWithColorbar(img)
    M,N = img.shape
    enlarged = myNearestNeighborInterpolation(img,3*M-2,2*N-1)
    imageWithColorbar(enlarged)    

def part_d(img):
    ##################### Part(d) ########################
    imageWithColorbar(img)
    M,N = img.shape
    enlarged = myBicubicInterpolation(img,3*M-2,2*N-1)
    imageWithColorbar(enlarged) 

def part_e(img):
    ##################### Part(e) ########################
    M,N = img.shape
    imageWithColorbar(img,'jet')
    imageWithColorbar(myBilinearInterpolation(img,5*M-4,4*N-3),'jet')
    imageWithColorbar(myNearestNeighborInterpolation(img,5*M-4,4*N-3),'jet')
    imageWithColorbar(myBicubicInterpolation(img,5*M-4,4*N-3),'jet')

def part_f(img):
    ##################### Part(f) ########################
    imageWithColorbar(img)
    rotated = myImageRotation(img, math.pi/6)
    imageWithColorbar(rotated) 

if __name__=='__main__':
        
    circles = mpimg.imread('../data/circles_concentric.png')
    barbara = mpimg.imread('../data/barbaraSmall.png')
    
    part_a(circles)
    part_b(barbara)
    part_c(barbara)
    part_d(barbara)
    part_e(barbara[:20,:20])
    part_f(barbara)

