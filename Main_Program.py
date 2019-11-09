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
Na = 50     # Number of Airfoil Points

# Wing Parameter
b = 2.0         # Wing Span
Lambda = 0.8    # Taper Ratio
theta = 10.0    # Sweep Angle in Degree
cr = 1.0        # Chord Root
Np = 11         # Number of Airfoil Section (Odd)

ct = Lambda*cr # Chord Tip
theta = theta*math.pi/180.0 # Convert Sweep Angle to radian

# 1. Create Airfoil
xa,ya = Airfoil_Generator.NACA4Digit(Na,m,p,t)

# 2. Create Wing and Panel
xw,yw,zw = Wing_Generator.Geo_Wing(xa,ya,Np,b,cr,ct,theta)
