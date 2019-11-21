from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import math
from matplotlib import cm
import matplotlib.pyplot as plt

def Geo_Wing(xa,ya,Np,b,cr,ct,theta):
    
    # Airfoil Parameter 
    Na = xa.shape[0]

    # Wing Coordinate
    xw = np.zeros((Na,Np))
    yw = np.zeros((Na,Np))
    zw = np.zeros((Na,Np))

    # Root Airfoil
    icr = int((Np-1)/2)
    for i in range(Na) :
        xw[i][icr] = xa[i]*cr 
        zw[i][icr] = ya[i]*cr
        yw[i][icr] = 0.0 

    # Wing Geometry
    b2 = b/2      # Semispan
    bl = b/(Np-1) # Airfoil Space
    for i in range(icr) :
        c =  (i*bl*cr+(b2-i*bl)*ct)/b2  # Airfoil Chord
        for j in range(Na) :
            # Left Wing
            xw[j][i] = xa[j]*c + ((icr-i)/icr)*b2*math.tan(theta)
            zw[j][i] = ya[j]*c
            yw[j][i] = -b2 + i*bl
            # Right Wing
            xw[j][Np-1-i] = xw[j][i]
            zw[j][Np-1-i] = zw[j][i]
            yw[j][Np-1-i] = -yw[j][i]
    

    # Plot
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(xw, yw, zw)
    # ax.plot_surface(xw, yw, zw,rstride=1,cstride=1,alpha=0,linewidth=0.5, edgecolors='b')

    ax.set_xlim3d(0,1)
    ax.set_ylim3d(-b2,b2)
    ax.set_zlim3d(-0.2,0.2)   
    plt.title('3D Wing Geometry') 
    plt.savefig('output/wing_geo.png')

    # Return Values
    return xw,yw,zw    


def Panel_Wing(Na,Np,x,y,z):

    # Panel Surface Area and Normal Vector
    S = np.zeros((Na-1,Np-1))     # Panel Surface Area
    nx = np.zeros((Na-1,Np-1))    # Normal Vector x-component
    ny = np.zeros((Na-1,Np-1))    # Normal Vector y-component
    nz = np.zeros((Na-1,Np-1))    # Normal Vector z-component

    for i in range(Na-1) :
        for j in range(Np-1) :
            # Diagonal Vector
            A = [(x[i+1][j+1]-x[i][j]), (y[i+1][j+1]-y[i][j]), (z[i+1][j+1]-z[i][j])]
            B = [(x[i+1][j]-x[i][j+1]), (y[i+1][j]-y[i][j+1]), (z[i+1][j]-z[i][j+1])]
            # Panel Surface Area
            n = np.cross(A,B)
            mag_n = np.linalg.norm(n)
            S[i][j] = mag_n/2
            # Normal Vector
            nx[i][j] = -n[0]/mag_n 
            ny[i][j] = -n[1]/mag_n
            nz[i][j] = -n[2]/mag_n
    
    # Collocation Points (C)
    cx = np.zeros((Na-1,Np-1))
    cy = np.zeros((Na-1,Np-1))
    cz = np.zeros((Na-1,Np-1))
    for i in range(Na-1) :
        for j in range(Np-1) :
            cx[i][j] = (x[i][j] + x[i][j+1] + x[i+1][j] + x[i+1][j+1])/4
            cy[i][j] = (y[i][j] + y[i][j+1] + y[i+1][j] + y[i+1][j+1])/4
            cz[i][j] = (z[i][j] + z[i][j+1] + z[i+1][j] + z[i+1][j+1])/4
    
    # Vector u,p,o
    # Unit vector in longitudinal (u)
    ux = np.zeros((Na-1,Np-1))    # x-component
    uy = np.zeros((Na-1,Np-1))    # y-component
    uz = np.zeros((Na-1,Np-1))    # z-component
    # Unit vector in transversal (p)
    px = np.zeros((Na-1,Np-1))    # x-component
    py = np.zeros((Na-1,Np-1))    # y-component
    pz = np.zeros((Na-1,Np-1))    # z-component
    # Unit vector prependicular (n x u) (o)
    ox = np.zeros((Na-1,Np-1))    # x-component
    oy = np.zeros((Na-1,Np-1))    # y-component
    oz = np.zeros((Na-1,Np-1))    # z-component
    for i in range(Na-1) :
        for j in range(Np-1) :
            # Vector u
            u = np.zeros(3)
            u[0] = (x[i+1][j] + x[i+1][j+1] - x[i][j] - x[i][j+1])/2
            u[1] = (y[i+1][j] + y[i+1][j+1] - y[i][j] - y[i][j+1])/2
            u[2] = (z[i+1][j] + z[i+1][j+1] - z[i][j] - z[i][j+1])/2
            mag_u = np.linalg.norm(u)
            ux[i][j] = u[0]/mag_u
            uy[i][j] = u[1]/mag_u
            uz[i][j] = u[2]/mag_u
            # Vector p
            p = np.zeros(3)
            p[0] = (x[i][j+1] + x[i+1][j+1] - x[i+1][j] - x[i][j])/2
            p[1] = (y[i][j+1] + y[i+1][j+1] - y[i+1][j] - y[i][j])/2
            p[2] = (z[i][j+1] + z[i+1][j+1] - z[i+1][j] - z[i][j])/2
            mag_p = np.linalg.norm(p)
            px[i][j] = p[0]/mag_p
            py[i][j] = p[1]/mag_p
            pz[i][j] = p[2]/mag_p
            # vector o (n x u)
            ox[i][j] = ny[i][j]*uz[i][j] - nz[i][j]*uy[i][j]
            oy[i][j] = -nx[i][j]*uz[i][j] + nz[i][j]*ux[i][j] 
            oz[i][j] = nx[i][j]*uy[i][j] - ny[i][j]*uz[i][j]
    
    # Local Panel Coordinate and Panel Side Length
    # Local Panel Coordinate (x,y)
    x1 = np.zeros((Na-1,Np-1))
    y1 = np.zeros((Na-1,Np-1))
    x2 = np.zeros((Na-1,Np-1))
    y2 = np.zeros((Na-1,Np-1))
    x3 = np.zeros((Na-1,Np-1))
    y3 = np.zeros((Na-1,Np-1))
    x4 = np.zeros((Na-1,Np-1))
    y4 = np.zeros((Na-1,Np-1))
    # Panel Side Length
    d1 = np.zeros((Na-1,Np-1))
    d2 = np.zeros((Na-1,Np-1))
    d3 = np.zeros((Na-1,Np-1))
    d4 = np.zeros((Na-1,Np-1))
    for i in range(Na-1) :
        for j in range(Np-1) :
            x1[i][j] = (x[i][j]-cx[i][j])*ux[i][j]+(y[i][j]-cy[i][j])*uy[i][j]+(z[i][j]-cz[i][j])*uz[i][j]
            y1[i][j] = (x[i][j]-cx[i][j])*ox[i][j]+(y[i][j]-cy[i][j])*oy[i][j]+(z[i][j]-cz[i][j])*oz[i][j]
            
            x2[i][j] = (x[i+1][j]-cx[i][j])*ux[i][j]+(y[i+1][j]-cy[i][j])*uy[i][j]+(z[i+1][j]-cz[i][j])*uz[i][j]
            y2[i][j] = (x[i+1][j]-cx[i][j])*ox[i][j]+(y[i+1][j]-cy[i][j])*oy[i][j]+(z[i+1][j]-cz[i][j])*oz[i][j]
            
            x3[i][j] = (x[i+1][j+1]-cx[i][j])*ux[i][j]+(y[i+1][j+1]-cy[i][j])*uy[i][j]+(z[i+1][j+1]-cz[i][j])*uz[i][j]
            y3[i][j] = (x[i+1][j+1]-cx[i][j])*ox[i][j]+(y[i+1][j+1]-cy[i][j])*oy[i][j]+(z[i+1][j+1]-cz[i][j])*oz[i][j]
            
            x4[i][j] = (x[i][j+1]-cx[i][j])*ux[i][j]+(y[i][j+1]-cy[i][j])*uy[i][j]+(z[i][j+1]-cz[i][j])*uz[i][j]
            y4[i][j] = (x[i][j+1]-cx[i][j])*ox[i][j]+(y[i][j+1]-cy[i][j])*oy[i][j]+(z[i][j+1]-cz[i][j])*oz[i][j]

            d1[i][j] = math.sqrt((x2[i][j]-x1[i][j])**2+(y2[i][j]-y1[i][j])**2)
            d2[i][j] = math.sqrt((x3[i][j]-x2[i][j])**2+(y3[i][j]-y2[i][j])**2)
            d3[i][j] = math.sqrt((x4[i][j]-x3[i][j])**2+(y4[i][j]-y3[i][j])**2)
            d4[i][j] = math.sqrt((x1[i][j]-x4[i][j])**2+(y1[i][j]-y4[i][j])**2)

    # # Plot (Check Normal Vector)
    # fig = plt.figure(2)
    # ax = fig.gca(projection='3d')
    # ax.quiver(cx, cy, cz, nx, ny, nz, length=0.05, normalize=True)
    # plt.show()
    
    # Resize Panel Parameter
    numpanel = (Na-1)*(Np-1)
    np.resize(x1,(1,numpanel))
    np.resize(x2,(1,numpanel))
    np.resize(x3,(1,numpanel))
    np.resize(x4,(1,numpanel))
    np.resize(y1,(1,numpanel))
    np.resize(y2,(1,numpanel))
    np.resize(y3,(1,numpanel))
    np.resize(y4,(1,numpanel))

    np.resize(nx,(1,numpanel))
    np.resize(ny,(1,numpanel))
    np.resize(nz,(1,numpanel))

    np.resize(cx,(1,numpanel))
    np.resize(cy,(1,numpanel))
    np.resize(cz,(1,numpanel))
    
    np.resize(ux,(1,numpanel))
    np.resize(uy,(1,numpanel))
    np.resize(uz,(1,numpanel))

    np.resize(ox,(1,numpanel))
    np.resize(oy,(1,numpanel))
    np.resize(oz,(1,numpanel))

    np.resize(S,(1,numpanel))
    # Local Coordinate    
    X = np.zeros(numpanel,numpanel)
    Y = np.zeros(numpanel,numpanel)
    Z = np.zeros(numpanel,numpanel)
    for i in range(numpanel) :
        for j in range(numpanel) :
            X[i,j] = (x[j]-cx[i])*ux[i]+(y[j]-cy[i])*uy[i]+(z[j]-cz[i])*uz[i]
            Y[i,j] = (x[j]-cx[i])*ox[i]+(y[j]-cy[i])*oy[i]+(z[j]-cz[i])*oz[i]
            Z[i,j] = (x[j]-cx[i])*nx[i]+(y[j]-cy[i])*ny[i]+(z[j]-cz[i])*nz[i]

    # Return Values 
    return x1,x2,x3,x4,y1,y2,y3,y4,S,n,X,Y,Z
