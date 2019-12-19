# import libraries and modules needed
import numpy
from scipy import integrate, linalg
from matplotlib import pyplot

class Freestream:
    """
    Freestream conditions.
    """
    def __init__(self, u_inf = 1.0, alpha =0.0):
        """
        Sets the freestream speed and angle (in degrees).
        
        Parameters
        ----------
        u_inf: float, optional
            Freestream speed;
            default: 1.0.
        alpha: float, optional
            Angle of attack in degrees;
            default 0.0.
        """
        self.u_inf = u_inf
        self.alpha = numpy.radians(alpha)  # degrees to radians
        
        self.vector = numpy.array([self.u_inf*numpy.cos(self.alpha),
                                   0, 
                                   self.u_inf*numpy.sin(self.alpha) ])
                                   
class Panel:
    """
    Panel object storing panel properties
    """
    def __init__(self, x1, x2, x3, x4, y1, y2, y3, y4, S, nx, ny, nz):
        """
        Parameters
        ----------
        x1, x2, x3, x4: float
            x-coordinate (local) of the corner points
        y1, y2, y3, y4: float
            y-coordinate (local) of the corner points    
        S: float
            Panel surface area
        nx, ny, nz: float
            x, y, and z component of normal vector at each panel
        normal: 1D array
            panel outward normal
        sigma: float
            source strength
        myu: float
            doublet strength
        """
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        
        self.S = S
        self.normal = numpy.array([nx, ny, nz])
        
        self.sigma = 0
        self.myu = 0

        # only useful for wake
        self.ind_upper = 99999
        self.ind_lower = 199999

def calculate_sigma(panels, freestream):
    """
    Calculate the source strength for each panel in panels.
    
    Parameters
    ----------
    panels: 1D array of panel objects
        List of panels.
    freestream: object
        Object defining freestream
    """
    for i, panel_i in enumerate(panels):
        panels[i].sigma = freestream.vector @ panel_i.normal

# computing panel_i influence to OTHER panels
# input : panels.x1,x2,x3,x4, y1,y2,y3,y4, Xoftherpanels, Yofotherpanels,  
# source # doublet
def influence_coeff(panels, X, Y, Z):
    """
    Builds the source & doublet contribution matrix for the potential value at each other panel.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    X: 2D array of floats [i,j]
        x-coordinate of panel j with respect to local coordinate panel i
    Y: 2D array of floats [i,j]
        y-coordinate of panel j with respect to local coordinate panel i
    Z: 2D array of floats [i,j]
        z-coordinate of panel j with respect to local coordinate panel i
        
    Returns
    -------
    A: 2D Numpy array of floats
        Doublet contribution matrix.
    B: 2D Numpy array of floats
        Source contribution matrix.
    """
    A = numpy.empty((panels.size, panels.size), dtype=float)
    B = numpy.empty((panels.size, panels.size), dtype=float)
    
    for i, panel_i in enumerate(panels):
        # compute the PROPERTY of PANEL_i
        y21 = panel_i.y2 - panel_i.y1
        y32 = panel_i.y3 - panel_i.y2
        y43 = panel_i.y4 - panel_i.y3
        y14 = panel_i.y1 - panel_i.y4
        
        x21 = panel_i.x2 - panel_i.x1
        x32 = panel_i.x3 - panel_i.x2
        x43 = panel_i.x4 - panel_i.x3
        x14 = panel_i.x1 - panel_i.x4        
        
        d12 = numpy.sqrt(x21**2 + y21**2)
        d23 = numpy.sqrt(x32**2 + y32**2)
        d34 = numpy.sqrt(x43**2 + y43**2)
        d41 = numpy.sqrt(x14**2 + y14**2)
        
        m12 = y21 / x21
        m23 = y32 / x32
        m34 = y43 / x43
        m41 = y14 / x14
        
        # compute r, e , and h to other panels from panel_i       
        e1 = (X[i,:] - panel_i.x1)**2 + Z[i,:]**2
        e2 = (X[i,:] - panel_i.x2)**2 + Z[i,:]**2
        e3 = (X[i,:] - panel_i.x3)**2 + Z[i,:]**2
        e4 = (X[i,:] - panel_i.x4)**2 + Z[i,:]**2
        
        r = numpy.sqrt( X[i,:]**2 + Y[i,:]**2 + Z[i,:]**2 )
        r1 = numpy.sqrt( e1 + (Y[i,:] - panel_i.y1)**2 )
        r2 = numpy.sqrt( e2 + (Y[i,:] - panel_i.y2)**2 )
        r3 = numpy.sqrt( e3 + (Y[i,:] - panel_i.y3)**2 )
        r4 = numpy.sqrt( e4 + (Y[i,:] - panel_i.y4)**2 )
        
        h1 = (X[i,:] - panel_i.x1) * (Y[i,:] - panel_i.y1)
        h2 = (X[i,:] - panel_i.x2) * (Y[i,:] - panel_i.y2)
        h3 = (X[i,:] - panel_i.x3) * (Y[i,:] - panel_i.y3)
        h4 = (X[i,:] - panel_i.x4) * (Y[i,:] - panel_i.y4)
        
        # now compute the influence coefficient
        for j in range(len(panels)):
        #for j, panel_j in enumerate(panels):    
            if j != i:                                          # not the same panel
                if r[j] >= ( 5 * numpy.sqrt(panel_i.S) ):       # far-field
                    A[j,i] = -panel_i.S * Z[i,j] * r[j]**(-3)
                    B[j,i] = -panel_i.S / r[j]
                else:
                    A[j,i] = ( numpy.arctan2(m12*e1[j] - h1[j], Z[i,j]*r1[j])
                             - numpy.arctan2(m12*e2[j] - h2[j], Z[i,j]*r2[j])
                             + numpy.arctan2(m23*e2[j] - h2[j], Z[i,j]*r2[j])
                             - numpy.arctan2(m23*e3[j] - h3[j], Z[i,j]*r3[j])
                             + numpy.arctan2(m34*e3[j] - h3[j], Z[i,j]*r3[j])
                             - numpy.arctan2(m34*e4[j] - h4[j], Z[i,j]*r4[j])
                             + numpy.arctan2(m41*e4[j] - h4[j], Z[i,j]*r4[j])
                             - numpy.arctan2(m41*e1[j] - h1[j], Z[i,j]*r1[j]) )
                    B[j,i] = -(((X[i,j] - panel_i.x1)*y21 - (Y[i,j] - panel_i.y1)*x21) 
                                / d12 * numpy.log( (r1[j]+r2[j]+d12)/(r1[j]+r2[j]-d12) ) 
                             + ((X[i,j] - panel_i.x2)*y32 - (Y[i,j] - panel_i.y2)*x32) 
                                / d23 * numpy.log( (r2[j]+r3[j]+d23)/(r2[j]+r3[j]-d23) )
                             + ((X[i,j] - panel_i.x3)*y43 - (Y[i,j] - panel_i.y3)*x43) 
                                / d34 * numpy.log( (r3[j]+r4[j]+d34)/(r3[j]+r4[j]-d34) )
                             + ((X[i,j] - panel_i.x4)*y14 - (Y[i,j] - panel_i.y4)*x14) 
                                / d41 * numpy.log( (r4[j]+r1[j]+d41)/(r4[j]+r1[j]-d41) ) 
                             - numpy.absolute(Z[i,j]) * A[j,i] )     
        A[i,i] = 0.5 * (4*numpy.pi)
        B[i,i] = -(((X[i,i] - panel_i.x1)*y21 - (Y[i,i] - panel_i.y1)*x21) 
                    / d12 * numpy.log( (r1[i]+r2[i]+d12)/(r1[i]+r2[i]-d12) ) 
                 + ((X[i,i] - panel_i.x2)*y32 - (Y[i,i] - panel_i.y2)*x32) 
                    / d23 * numpy.log( (r2[i]+r3[i]+d23)/(r2[i]+r3[i]-d23) )
                 + ((X[i,i] - panel_i.x3)*y43 - (Y[i,i] - panel_i.y3)*x43) 
                    / d34 * numpy.log( (r3[i]+r4[i]+d34)/(r3[i]+r4[i]-d34) )
                 + ((X[i,i] - panel_i.x4)*y14 - (Y[i,i] - panel_i.y4)*x14) 
                    / d41 * numpy.log( (r4[i]+r1[i]+d41)/(r4[i]+r1[i]-d41) ) )  
                   
    A = A / (4*numpy.pi)
    B = B / (4*numpy.pi)
    
    return A, B

# computing WAKE_panel_i influence to OTHER panels
# input : panels.x1,x2,x3,x4, y1,y2,y3,y4, Xoftherpanels, Yofotherpanels, Zofotherpanels, 
# source # doublet
def wake_influence_coeff(panels, X, Y, Z, size_of_wing_panels):
    """
    Builds the source contribution matrix for the potential value at each other panel.
    
    Parameters
    ----------
    panels: 1D array of Wake_Panel objects
        List of wake_panels.
    X: 2D array of floats [i,j] size [number of wake panels x number of panels]
        x-coordinate of panel j with respect to local coordinate wake_panel i
    Y: 2D array of floats [i,j]
        y-coordinate of panel j with respect to local coordinate wake_panel i
    Z: 2D array of floats [i,j]
        z-coordinate of panel j with respect to local coordinate wake_panel i
        
    Returns
    -------
    Aw: 2D Numpy array of floats
        Wake doublet contribution matrix.
    """
    Aw = numpy.empty((size_of_wing_panels, size_of_wing_panels), dtype=float)
    
    for i, panel_i in enumerate(panels):
        # compute the PROPERTY of PANEL_i
        y21 = panel_i.y2 - panel_i.y1
        y32 = panel_i.y3 - panel_i.y2
        y43 = panel_i.y4 - panel_i.y3
        y14 = panel_i.y1 - panel_i.y4
        
        x21 = panel_i.x2 - panel_i.x1
        x32 = panel_i.x3 - panel_i.x2
        x43 = panel_i.x4 - panel_i.x3
        x14 = panel_i.x1 - panel_i.x4        
        
        d12 = numpy.sqrt(x21**2 + y21**2)
        d23 = numpy.sqrt(x32**2 + y32**2)
        d34 = numpy.sqrt(x43**2 + y43**2)
        d41 = numpy.sqrt(x14**2 + y14**2)
        
        m12 = y21 / x21
        m23 = y32 / x32
        m34 = y43 / x43
        m41 = y14 / x14
        
        # compute r, e , and h to other panels from panel_i       
        e1 = (X[i,:] - panel_i.x1)**2 + Z[i,:]**2
        e2 = (X[i,:] - panel_i.x2)**2 + Z[i,:]**2
        e3 = (X[i,:] - panel_i.x3)**2 + Z[i,:]**2
        e4 = (X[i,:] - panel_i.x4)**2 + Z[i,:]**2
        
        r = numpy.sqrt( X[i,:]**2 + Y[i,:]**2 + Z[i,:]**2 )
        r1 = numpy.sqrt( e1 + (Y[i,:] - panel_i.y1)**2 )
        r2 = numpy.sqrt( e2 + (Y[i,:] - panel_i.y2)**2 )
        r3 = numpy.sqrt( e3 + (Y[i,:] - panel_i.y3)**2 )
        r4 = numpy.sqrt( e4 + (Y[i,:] - panel_i.y4)**2 )
        
        h1 = (X[i,:] - panel_i.x1) * (Y[i,:] - panel_i.y1)
        h2 = (X[i,:] - panel_i.x2) * (Y[i,:] - panel_i.y2)
        h3 = (X[i,:] - panel_i.x3) * (Y[i,:] - panel_i.y3)
        h4 = (X[i,:] - panel_i.x4) * (Y[i,:] - panel_i.y4)
        
        # now compute the influence coefficient
        for j in range(size_of_wing_panels):
        # NOW THERE ARE NO SAME PANEL AND NO FARFIELD
        #for j, panel_j in enumerate(panels):    
            coeff = ( numpy.arctan2(m12*e1[j] - h1[j], Z[i,j]*r1[j])
                     - numpy.arctan2(m12*e2[j] - h2[j], Z[i,j]*r2[j])
                     + numpy.arctan2(m23*e2[j] - h2[j], Z[i,j]*r2[j])
                     - numpy.arctan2(m23*e3[j] - h3[j], Z[i,j]*r3[j])
                     + numpy.arctan2(m34*e3[j] - h3[j], Z[i,j]*r3[j])
                     - numpy.arctan2(m34*e4[j] - h4[j], Z[i,j]*r4[j])
                     + numpy.arctan2(m41*e4[j] - h4[j], Z[i,j]*r4[j])
                     - numpy.arctan2(m41*e1[j] - h1[j], Z[i,j]*r1[j]) )
            Aw[j, panel_i.ind_upper] = coeff
            Aw[j, panel_i.ind_lower] = -coeff
                
    Aw = Aw / (4*numpy.pi)
    
    return Aw