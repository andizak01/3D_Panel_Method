from mpl_toolkits import mplot3d
import numpy as np 
import math
import matplotlib.pyplot as plt

import Airfoil_Generator
import Wing_Generator

# Global Input Parameter
mu = 1.0    # Kinematic Viscosity
U_inf = 1.0 # Freestream Velocity
alpha = 5.0 # AoA in Degree

alpha = alpha*math.pi/180.0 # Convert AoA to radian

# Airfoil Parameter (NACA 4-digit)
m = 2/100   # Maximum Camber
p = 40/100  # Location of Maximum Camber
t = 12/100  # Maximum Thickness
Na = 51     # Number of Airfoil Points (Odd) (1==Na)

# Wing Parameter
b = 2.0         # Wing Span
Lambda = 0.8    # Taper Ratio
theta = 10.0    # Sweep Angle in Degree
cr = 1.0        # Chord Root
Np = 11         # Number of Airfoil Section (Odd)

ct = Lambda*cr # Chord Tip
theta = theta*math.pi/180.0 # Convert Sweep Angle to radian

# 1. Create Airfoil
xa,ya = Airfoil_Generator.NACA4Digit(Na,m,p,t)              # Output : Airfoil Coordinate (x/c,y/c)

# 2. Create Wing and Panel
xw,yw,zw = Wing_Generator.Geo_Wing(xa,ya,Np,b,cr,ct,theta)  # Output : Wing Coordinate (x,y,z)
x1,x2,x3,x4,y1,y2,y3,y4,S,n,X,Y,Z = Wing_Generator.Panel_Wing(Na,Np,xw,yw,zw) # Output : Panel Side Length (d),Local Panel Coordinate (x,y), and Panel Area (S)
        
