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
    
    # Wake Geometry
    xwake = np.zeros(Np)
    ywake = np.zeros(Np)
    zwake = np.zeros(Np)
    for i in range(Np) :
        c =  (i*bl*cr+(b2-i*bl)*ct)/b2  # Airfoil Chord

        xwake[i] = xw[0][i] + 50.0*c
        ywake[i] = yw[0][i]
        zwake[i] = zw[0][i]

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
    return xw,yw,zw,xwake,ywake,zwake    


def Panel_Wing(Na,Np,x,y,z,xwake,ywake,zwake):

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
    
    # Wake Surface Area and Normal Vector
    Swake = np.zeros(Np-1)
    nxwake = np.zeros(Np-1)
    nywake = np.zeros(Np-1)
    nzwake = np.zeros(Np-1)
    for i in range(Np-1) :
        # Diagonal Vector
        A1 = [(x[0][i+1]-xwake[i]), (y[0][i+1]-ywake[i]), (z[0][i+1]-zwake[i])]
        B1 = [(x[0][i]-xwake[i+1]), (y[0][i]-ywake[i+1]), (z[0][i]-zwake[i+1])]
        # Panel Surface Area
        n1 = np.cross(A1,B1)
        mag_n1 = np.linalg.norm(n1)
        Swake[i] = mag_n1/2
        # Normal Vector
        nxwake[i] = n1[0]/mag_n1 
        nywake[i] = n1[1]/mag_n1
        nzwake[i] = n1[2]/mag_n1
    
    # Collocation Points (C)
    cx = np.zeros((Na-1,Np-1))
    cy = np.zeros((Na-1,Np-1))
    cz = np.zeros((Na-1,Np-1))
    for i in range(Na-1) :
        for j in range(Np-1) :
            cx[i][j] = (x[i][j] + x[i][j+1] + x[i+1][j] + x[i+1][j+1])/4
            cy[i][j] = (y[i][j] + y[i][j+1] + y[i+1][j] + y[i+1][j+1])/4
            cz[i][j] = (z[i][j] + z[i][j+1] + z[i+1][j] + z[i+1][j+1])/4
    
    # Wake Collocation Points (C)
    cxwake = np.zeros(Np-1)
    cywake = np.zeros(Np-1)
    czwake = np.zeros(Np-1)
    for i in range(Np-1) :
        cxwake[i] = (xwake[i]+ xwake[i+1] + x[0][i] + x[0][i+1])/4
        cywake[i] = (ywake[i]+ ywake[i+1] + y[0][i] + y[0][i+1])/4
        czwake[i] = (zwake[i]+ zwake[i+1] + z[0][i] + z[0][i+1])/4
    
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
    
    # Wake vector u,p,o
    # Unit vector in longitudinal (u)
    uxwake = np.zeros(Np-1)  # x-component
    uywake = np.zeros(Np-1)   # y-component
    uzwake = np.zeros(Np-1)    # z-component
    # Unit vector in transversal (p)
    pxwake = np.zeros(Np-1)   # x-component
    pywake = np.zeros(Np-1)    # y-component
    pzwake = np.zeros(Np-1)    # z-component
    # Unit vector prependicular (n x u) (o)
    oxwake = np.zeros(Np-1)   # x-component
    oywake = np.zeros(Np-1)    # y-component
    ozwake = np.zeros(Np-1)    # z-component
    for i in range(Np-1) :
        # Vector u
        u = np.zeros(3)
        u[0] = (x[0][i] + x[0][i+1] - xwake[i] - xwake[i+1])/2
        u[1] = (y[0][i] + y[0][i+1] - ywake[i] - ywake[i+1])/2
        u[2] = (z[0][i] + z[0][i+1] - zwake[i] - zwake[i+1])/2
        mag_u = np.linalg.norm(u)
        uxwake[i] = -u[0]/mag_u
        uywake[i] = -u[1]/mag_u
        uzwake[i] = -u[2]/mag_u
        # Vector p
        p = np.zeros(3)
        p[0] = (xwake[i+1] + x[0][i+1] - x[0][i] - xwake[i])/2
        p[1] = (ywake[i+1] + y[0][i+1] - y[0][i] - ywake[i])/2
        p[2] = (zwake[i+1] + z[0][i+1] - z[0][i] - zwake[i])/2
        mag_p = np.linalg.norm(p)
        pxwake[i] = p[0]/mag_p
        pywake[i] = p[1]/mag_p
        pzwake[i] = p[2]/mag_p
        # vector o (n x u)
        oxwake[i] = nywake[i]*uzwake[i] - nzwake[i]*uywake[i]
        oywake[i] = -nxwake[i]*uzwake[i] + nzwake[i]*uxwake[i] 
        ozwake[i] = nxwake[i]*uywake[i] - nywake[i]*uzwake[i]

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
            
            x4[i][j] = (x[i+1][j]-cx[i][j])*ux[i][j]+(y[i+1][j]-cy[i][j])*uy[i][j]+(z[i+1][j]-cz[i][j])*uz[i][j]
            y4[i][j] = (x[i+1][j]-cx[i][j])*ox[i][j]+(y[i+1][j]-cy[i][j])*oy[i][j]+(z[i+1][j]-cz[i][j])*oz[i][j]
            
            x3[i][j] = (x[i+1][j+1]-cx[i][j])*ux[i][j]+(y[i+1][j+1]-cy[i][j])*uy[i][j]+(z[i+1][j+1]-cz[i][j])*uz[i][j]
            y3[i][j] = (x[i+1][j+1]-cx[i][j])*ox[i][j]+(y[i+1][j+1]-cy[i][j])*oy[i][j]+(z[i+1][j+1]-cz[i][j])*oz[i][j]
            
            x2[i][j] = (x[i][j+1]-cx[i][j])*ux[i][j]+(y[i][j+1]-cy[i][j])*uy[i][j]+(z[i][j+1]-cz[i][j])*uz[i][j]
            y2[i][j] = (x[i][j+1]-cx[i][j])*ox[i][j]+(y[i][j+1]-cy[i][j])*oy[i][j]+(z[i][j+1]-cz[i][j])*oz[i][j]

            d1[i][j] = math.sqrt((x2[i][j]-x1[i][j])**2+(y2[i][j]-y1[i][j])**2)
            d2[i][j] = math.sqrt((x3[i][j]-x2[i][j])**2+(y3[i][j]-y2[i][j])**2)
            d3[i][j] = math.sqrt((x4[i][j]-x3[i][j])**2+(y4[i][j]-y3[i][j])**2)
            d4[i][j] = math.sqrt((x1[i][j]-x4[i][j])**2+(y1[i][j]-y4[i][j])**2)

    # Wake Local Panel Coordinate and Panel Side Length
    # Local Panel Coordinate (x,y)
    x1w = np.zeros(Np-1)
    y1w = np.zeros(Np-1)
    x2w = np.zeros(Np-1)
    y2w = np.zeros(Np-1)
    x3w = np.zeros(Np-1)
    y3w = np.zeros(Np-1)
    x4w = np.zeros(Np-1)
    y4w = np.zeros(Np-1)
    # Panel Side Length
    d1w = np.zeros(Np-1)
    d2w = np.zeros(Np-1)
    d3w = np.zeros(Np-1)
    d4w = np.zeros(Np-1)
    for i in range(Np-1) :
        
        x4w[i] = (xwake[i]-cxwake[i])*uxwake[i]+(ywake[i]-cywake[i])*uywake[i]+(zwake[i]-czwake[i])*uzwake[i]
        y4w[i] = (xwake[i]-cxwake[i])*oxwake[i]+(ywake[i]-cywake[i])*oywake[i]+(zwake[i]-czwake[i])*ozwake[i]
            
        x1w[i] = (x[0][i]-cxwake[i])*uxwake[i]+(y[0][i]-cywake[i])*uywake[i]+(z[0][i]-czwake[i])*uzwake[i]
        y1w[i] = (x[0][i]-cxwake[i])*oxwake[i]+(y[0][i]-cywake[i])*oywake[i]+(z[0][i]-czwake[i])*ozwake[i]

        x2w[i] = (x[0][i+1]-cxwake[i])*uxwake[i]+(y[0][i+1]-cywake[i])*uywake[i]+(z[0][i+1]-czwake[i])*uzwake[i]
        y2w[i] = (x[0][i+1]-cxwake[i])*oxwake[i]+(y[0][i+1]-cywake[i])*oywake[i]+(z[0][i+1]-czwake[i])*ozwake[i]

        x3w[i] = (xwake[i+1]-cxwake[i])*uxwake[i]+(ywake[i+1]-cywake[i])*uywake[i]+(zwake[i+1]-czwake[i])*uzwake[i]
        y3w[i] = (xwake[i+1]-cxwake[i])*oxwake[i]+(ywake[i+1]-cywake[i])*oywake[i]+(zwake[i+1]-czwake[i])*ozwake[i]

        d1w[i] = math.sqrt((x2w[i]-x1w[i])**2+(y2w[i]-y1w[i])**2)
        d2w[i] = math.sqrt((x3w[i]-x2w[i])**2+(y3w[i]-y2w[i])**2)
        d3w[i] = math.sqrt((x4w[i]-x3w[i])**2+(y4w[i]-y3w[i])**2)
        d4w[i] = math.sqrt((x1w[i]-x4w[i])**2+(y1w[i]-y4w[i])**2)
    
    # # Plot (Check Vector)
    # fig = plt.figure(2)
    # ax = fig.gca(projection='3d')
    # ax.quiver(cx, cy, cz, nx, ny, nz, length=0.05, normalize=True)
    # ax.quiver(cx, cy, cz, ux, uy, uz, length=0.05, normalize=True)
    # ax.quiver(cx, cy, cz, ox, oy, oz, length=0.05, normalize=True)
    # plt.show()
    
    # Resize Panel Parameter
    numpanel = (Na-1)*(Np-1)
    x1 = np.reshape(x1,(-1))
    x2 = np.reshape(x2,(-1))
    x3 = np.reshape(x3,(-1))
    x4 = np.reshape(x4,(-1))
    y1 = np.reshape(y1,(-1))
    y2 = np.reshape(y2,(-1))
    y3 = np.reshape(y3,(-1))
    y4 = np.reshape(y4,(-1))

    nx = np.reshape(nx,(-1))
    ny = np.reshape(ny,(-1))
    nz = np.reshape(nz,(-1))

    cx = np.reshape(cx,(-1))
    cy = np.reshape(cy,(-1))
    cz = np.reshape(cz,(-1))
    
    ux = np.reshape(ux,(-1))
    uy = np.reshape(uy,(-1))
    uz = np.reshape(uz,(-1))

    ox = np.reshape(ox,(-1))
    oy = np.reshape(oy,(-1))
    oz = np.reshape(oz,(-1))

    S = np.reshape(S,(-1))
   
    # Local Coordinate    
    X = np.zeros((numpanel,numpanel))
    Y = np.zeros((numpanel,numpanel))
    Z = np.zeros((numpanel,numpanel))
    for i in range(numpanel) :
        for j in range(numpanel) :
            # X[i][j] = (cx[0][j]-cx[0][i])*ux[0][i]+(cy[0][j]-cy[0][i])*uy[0][i]+(cz[0][j]-cz[0][i])*uz[0][i]
            # Y[i][j] = (cx[0][j]-cx[0][i])*ox[0][i]+(cy[0][j]-cy[0][i])*oy[0][i]+(cz[0][j]-cz[0][i])*oz[0][i]
            # Z[i][j] = (cx[0][j]-cx[0][i])*nx[0][i]+(cy[0][j]-cy[0][i])*ny[0][i]+(cz[0][j]-cz[0][i])*nz[0][i]
            X[i][j] = (cx[j]-cx[i])*ux[i]+(cy[j]-cy[i])*uy[i]+(cz[j]-cz[i])*uz[i]
            Y[i][j] = (cx[j]-cx[i])*ox[i]+(cy[j]-cy[i])*oy[i]+(cz[j]-cz[i])*oz[i]
            Z[i][j] = (cx[j]-cx[i])*nx[i]+(cy[j]-cy[i])*ny[i]+(cz[j]-cz[i])*nz[i]

    # Wake Local Coordinate    
    Xwake = np.zeros((Np-1,numpanel))
    Ywake = np.zeros((Np-1,numpanel))
    Zwake = np.zeros((Np-1,numpanel))
    for i in range(Np-1) :
        for j in range(numpanel) :
            
            Xwake[i][j] = (cx[j]-cxwake[i])*uxwake[i]+(cy[j]-cywake[i])*uywake[i]+(cz[j]-czwake[i])*uzwake[i]
            Ywake[i][j] = (cx[j]-cxwake[i])*oxwake[i]+(cy[j]-cywake[i])*oywake[i]+(cz[j]-czwake[i])*ozwake[i]
            Zwake[i][j] = (cx[j]-cxwake[i])*nxwake[i]+(cy[j]-cywake[i])*nywake[i]+(cz[j]-czwake[i])*nzwake[i]
    
    # Return Values 
    return cx,cy,cz,x1,x2,x3,x4,y1,y2,y3,y4,S,nx,ny,nz,X,Y,Z,x1w,x2w,x3w,x4w,y1w,y2w,y3w,y4w,Swake,Xwake,Ywake,Zwake

