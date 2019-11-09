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
    plt.show()

    # Return Values
    return xw,yw,zw    


def Panel_Wing(xw,yw,zw):

    # Variables
    S = np.zeros()  # Panel Area
    n = np.zeros()  # Normal Vector

    # Return Values
    return S,n
