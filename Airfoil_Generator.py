import numpy as np 
import math
import matplotlib.pyplot as plt

def NACA4Digit(Na,m,p,t):
    
    # x,y Coordinate
    x = np.zeros(Na)
    y = np.zeros(Na)

    for i in range(Na):
        x[i] = (1/2)*(math.cos((i/(Na-1))*2*math.pi)+1)
        if i == 0 or i==Na-1:
            y[i] = 0
        else :    
            yt = (t/0.2)*(0.2969*math.sqrt(x[i])-0.126*x[i]-0.35160*(x[i]**2)+0.2843*(x[i]**3)-0.1015*(x[i]**4))
            if x[i] < p :
                yc = (m/(p**2))*(2*p*x[i]-x[i]**2)
                dyc = (2*m/(p**2))*(p-x[i])

            else :
                yc = (m/((1-p)**2))*(1-2*p+2*p*x[i]-x[i]**2)
                dyc = (2*m/((1-p)**2))*(p-x[i])

            thetac = math.atan(dyc)

            if  i <= Na/2 :
                x[i] = x[i] + yt*math.sin(thetac)       
                y[i] = yc - yt*math.cos(thetac)
            else : 
                x[i] = x[i] - yt*math.sin(thetac)
                y[i] = yc + yt*math.cos(thetac)
    
    # Save Result
    plt.figure(0)
    plt.plot(x,y,'-o',linewidth=1)
    plt.ylim(-0.5, 0.5)
    plt.xlim(0,1)
    plt.title('Airfoil Section')
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.savefig('output/Airfoil.png')

    # Return Values
    return x,y    
