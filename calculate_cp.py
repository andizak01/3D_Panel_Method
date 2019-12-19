import numpy as np

def output(N_panel,Np,cx,cy,cz,panels,ux,uy,uz,px,py,pz,ox,oy,oz,nx,ny,nz,U_inf,Vx,Vy,Vz,S,Sw,q,b,cr):
    '''
    Input: N_panel: Numbers of panels
           Np: Number of airfoil section
           cx,cy,cz: collocation points
           panels: panel properties
           ux,uy,uz
           px,py,pz
           ox,oy,oz
           nx,ny,nz
           U_inf
           Vx,Vy,Vz
           S: area of each panels
           Sw: area of wing
           q: dynamic pressure
           b: wing span
           cr: chord root
    '''
    Fx = 0
    Fy = 0
    Fz = 0
    FL = 0
    FM = 0
    FN = 0
    ql = np.zeros((N_panel,1))
    qm = np.zeros((N_panel,1))
    qn = np.zeros((N_panel,1))
    qo = np.zeros((N_panel,1))
    gu = np.zeros((N_panel,1))
    go = np.zeros((N_panel,1))
    vx = np.zeros((N_panel,1))
    vy = np.zeros((N_panel,1))
    vz = np.zeros((N_panel,1))
    v = np.zeros((N_panel,1))
    cp = np.zeros((N_panel,1))
    
    #perhitungan ql, qm, qn
    for i in range(N_panel):
        #Perhitungan ql (chordwise velocity)
        if (N_panel-i) < (Np): #backward difference
            dx = cx[i]-cx[i-Np+1]
            dy = cy[i]-cy[i-Np+1]
            dz = cz[i]-cz[i-Np+1]
            dx2 = np.power(dx,2)
            dy2 = np.power(dy,2)
            dz2 = np.power(dz,2)
            dl = np.power((dx2+dy2+dz2),0.5)
            ql[i,0] = -1*(panels[i].myu-panels[i-Np+1].myu)/dl
        elif i < Np : #forward difference
            dx = cx[i+Np-1]-cx[i]
            dy = cy[i+Np-1]-cy[i]
            dz = cz[i+Np-1]-cz[i+Np-1]
            dx2 = np.power(dx,2)
            dy2 = np.power(dy,2)
            dz2 = np.power(dz,2)
            dl = np.power((dx2+dy2+dz2),0.5)
            ql[i,0] = -1*(panels[i+Np-1].myu-panels[i].myu)/dl
        else: #central difference
            dx = cx[i+Np-1]-cx[i]
            dy = cy[i+Np-1]-cy[i]
            dz = cz[i+Np-1]-cz[i]
            drupwind = np.power((np.power(dx,2)+np.power(dy,2)+np.power(dz,2)),0.5)
            dx = cx[i]-cx[i-Np+1]
            dy = cy[i]-cy[i-Np+1]
            dz = cz[i]-cz[i-Np+1]
            drdownwind= np.power((np.power(dx,2)+np.power(dy,2)+np.power(dz,2)),0.5)
            pu = np.polyfit( [-drdownwind,0,drupwind],[panels[i-Np+1].myu,panels[i].myu,panels[i+Np-1].myu],2)
            ql[i,0] = pu[1]
        #Perhiungan qm (spanwise velocity)
        if (i+1) % (Np-1) == 0: #backward difference
            dx = cx[i]-cx[i-1]
            dy = cy[i]-cy[i-1]
            dz = cz[i]-cz[i-1]
            dx2 = np.power(dx,2)
            dy2 = np.power(dy,2)
            dz2 = np.power(dz,2)
            dm = np.power((dx2+dy2+dz2),0.5)
            qm[i,0] = -1*(panels[i].myu-panels[i-1].myu)/dm
        elif i %(Np-1) == 0: #forward difference
            dx = cx[i+1]-cx[i]
            dy = cy[i+1]-cy[i]
            dz = cz[i+1]-cz[i]
            dx2 = np.power(dx,2)
            dy2 = np.power(dy,2)
            dz2 = np.power(dz,2)
            dm = np.power((dx2+dy2+dz2),0.5)
            qm[i,0] = -1*(panels[i+1].myu-panels[i].myu)/dm
        else:
            dx = cx[i+1]-cx[i]
            dy = cy[i+1]-cy[i]
            dz = cz[i+1]-cz[i]
            drupwind = np.power((np.power(dx,2)+np.power(dy,2)+np.power(dz,2)),0.5)
            dx = cx[i]-cx[i-1]
            dy = cy[i]-cy[i-1]
            dz = cz[i]-cz[i-1]
            drdownwind= np.power((np.power(dx,2)+np.power(dy,2)+np.power(dz,2)),0.5)
            pu = np.polyfit( [-drdownwind,0,drupwind],[panels[i-1].myu,panels[i].myu,panels[i+1].myu],2)
            qm[i,0] = pu[1]
        #Perhitungan qn (normal velocity)
        #qn[i,0] = panels[i].sigma
    
    #Perhitungan Qk dan Cp
    for i in range(N_panel):
        qo[i,0] = (px[i]*ox[i] + py[i]*oy[i] * pz[i]*oz[i]) * qm[i,0]
        gu[i,0] = ux[i]*Vx + uy[i]*Vy + uz[i]*Vz
        go[i,0] = ox[i]*Vx + oy[i]*Vy + oz[i]*Vz

    for i in range(N_panel):
        vx[i,0] = (-ql[i,0]+gu[i,0])*ux[i] + (-qo[i,0]+go[i,0])*ox[i]
        vy[i,0] = (-ql[i,0]+gu[i,0])*uy[i] + (-qo[i,0]+go[i,0])*oy[i]
        vz[i,0] = (-ql[i,0]+gu[i,0])*uz[i] + (-qo[i,0]+go[i,0])*oz[i]
        v[i,0]  = np.power((np.power(vx[i,0],2)+np.power(vy[i,0],2)+np.power(vz[i,0],2)),0.5)
        cp[i,0] = 1-np.power(v[i,0],2)/np.power(U_inf,2)
    
    for i in range(N_panel):
        dX = -cp[i,0]*S[i]*nx[i]*q
        dY = -cp[i,0]*S[i]*ny[i]*q
        dZ = -cp[i,0]*S[i]*nz[i]*q
        Fx = Fx + dX
        Fy = Fy + dY
        Fz = Fz + dZ
        #FL = FL - dY*cz[i] + dZ*cy[i]
        #FM = FM + dX*cz[i] - dZ*cx[i]
        #FN = FN - dX*cy[i] + dY*cx[i]
    num_row = int(N_panel/ (Np-1))
    num_col = Np-1
    cp = np.reshape(cp,(num_row,num_col))
    CX = Fx/q/Sw
    CY = Fy/q/Sw
    CZ = Fz/q/Sw
    #CL = FL/q/Sw/b
    #CM = FM/q/Sw/cr 
    
    return ql,qm,v,cp,CX,CY,CZ,ql,qo,gu,go