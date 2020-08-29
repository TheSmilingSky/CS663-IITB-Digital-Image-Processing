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

def part_d(barbara):
    ##################### Part(b) ########################
    imageWithColorbar(barbara)
    M,N = barbara.shape
    enlarged_barbara = myBicubicInterpolation(barbara,3*M-2,2*N-1)
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

if __name__=='__main__':
        
    circles = mpimg.imread('../data/circles_concentric.png')
    barbara = mpimg.imread('../data/barbaraSmall.png')
    
    # part_a(circles)
    # part_b(barbara)
    # part_c(barbara)
    # part_f(barbara)
    part_d(barbara)
    

