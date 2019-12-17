from mpl_toolkits import mplot3d
import numpy 
import math
import matplotlib.pyplot as plt

import Airfoil_Generator
import Wing_Generator

from funclass import Freestream, Panel, calculate_sigma, influence_coeff, wake_influence_coeff

# Global Input Parameter
mu = 1.0    # Kinematic Viscosity
U_inf = 1.0 # Freestream Velocity
alpha = 5.0 # AoA in Degree
alpha = alpha*math.pi/180.0 # Convert AoA to radian

# Airfoil Parameter (NACA 4-digit)
m = 0/100   # Maximum Camber
p = 0/100  # Location of Maximum Camber
t = 12/100  # Maximum Thickness
Na = 51     # Number of Airfoil Points (Odd) (1==Na)

# Wing Parameter
b = 2.0         # Wing Span
Lambda = 1.0    # Taper Ratio
theta = 0.0    # Sweep Angle in Degree
cr = 1.0        # Chord Root
Np = 11         # Number of Airfoil Section (Odd)

ct = Lambda*cr # Chord Tip
theta = theta*math.pi/180.0 # Convert Sweep Angle to radian

# 1. Create Airfoil
xa,ya = Airfoil_Generator.NACA4Digit(Na,m,p,t)              # Output : Airfoil Coordinate (x/c,y/c)

# 2. Create Wing and Panel
xw,yw,zw,xwake,ywake,zwake = Wing_Generator.Geo_Wing(xa,ya,Np,b,cr,ct,theta)  # Output : Wing Coordinate (x,y,z)
cx,cy,cz,x1,x2,x3,x4,y1,y2,y3,y4,S,nx,ny,nz,X,Y,Z, x1_wake,x2_wake,x3_wake,x4_wake,y1_wake,y2_wake,y3_wake,y4_wake,S_wake,X_wake,Y_wake,Z_wake = Wing_Generator.Panel_Wing(Na,Np,xw,yw,zw,xwake,ywake,zwake) 

# 3. Panel Objects, Influence Coefficients, Singularities Strength 
# 3.1 Wing Panels
# Number of panels for wing
N_panel = x1.size
panels = numpy.empty(N_panel, dtype=object)

# Create 'objects' of wing panels
for i in range(N_panel):
   panels[i] = Panel(x1[i], x2[i], x3[i], x4[i], y1[i], y2[i], y3[i], y4[i], S[i], nx[i], ny[i], nz[i])

# Computing Influence Coefficient of WING Panels
A, B = influence_coeff(panels, X, Y, Z)

# 3.2 Create wake panels
# Number of panels for wake
N_wake_panel = x1_wake.size #number of panels in the spanwise direction
wake_panels = numpy.empty(N_wake_panel, dtype=object)

# Create 'objects' of wake panels
ind_lower = range(Np)
ind_upper = range(N_panel-Np-2,N_panel)
for i in range(N_wake_panel):
   wake_panels[i] = Panel(x1_wake[i], x2_wake[i], x3_wake[i], x4_wake[i], y1_wake[i], y2_wake[i], y3_wake[i], y4_wake[i], S_wake[i], 0, 0, 1)
   wake_panels[i].ind_upper = ind_upper[i]
   wake_panels[i].ind_lower = ind_lower[i]

# Computing Influence Coefficient of WAKE Panels
Aw = wake_influence_coeff(wake_panels, X_wake, Y_wake, Z_wake, panels.size)

# 3.3 Construct the RHS and LHS
freestream = Freestream(U_inf,alpha)

calculate_sigma(panels, freestream)

# Compute the Right Hand Side
RHS = numpy.dot(-B, [panel.sigma for panel in panels])

# Compute the Left Hand Side Atot
Atot = A + Aw

# 3.4 Solve for DOUBLET strengths
myus = numpy.linalg.solve(Atot, RHS)

# 3.5 Store doublet strength on each panel
for i, panel in enumerate(panels):
   panels[i].myu = myus[i]

# COBACOBA PLOT
cx_mat = numpy.reshape(cx,(Na-1,Np-1))
cy_mat = numpy.reshape(cy,(Na-1,Np-1))
cz_mat = numpy.reshape(cz,(Na-1,Np-1)) 
panels_mat = numpy.reshape(panels,(Na-1,Np-1))

Np12 = int((Np-1)/2)
# print(cx_mat[:,Np12])

plt.figure(20)
plt.plot(cx_mat[:,Np12],[panel.myu for panel in panels_mat[:,Np12]] )
plt.show()

# print([panel.myu for panel in panels_mat[:,Np12]])