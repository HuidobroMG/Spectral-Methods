# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:14:29 2021

@author: huido
"""

import numpy as np
import matplotlib.pyplot as plt

Nvars = 1
Ncoefs = 20
Niters = 10
N = 200

#-----------------------------------------------------------------------------

#GRID
x = np.zeros(N)
w = np.zeros(N)

# GRID IN [-1, 1] SPACE
for i in range(N):
    x[i] = -np.cos(np.pi*i/(N-1)) # (CHEBYSHEV-GAUSS-LOBATTO NODES)
    w[i] = np.pi/N # (CHEBYSHEV-GAUSS-LOBATTO WEIGHTS)
w[0] /= 2
w[-1] /= 2

# COMPACTIFICATION OF REAL SPACE
# r = alpha/2*(1+x), r = [0, alpha]

alpha = 10
r = alpha/2*(1+x)
K = alpha/2

#-----------------------------------------------------------------------------

# FUNCTIONS

# QUADRATURE INTEGRATION
def CGL(f, w, N):
    return sum(f[i]*w[i] for i in range(N))

# INTEGRAL OF CHEBYSHEV POLYNOMIALS <Ti|Tj>
def TiTj(i, j):
    if i != j:
        return 0
    else:
        if i == 0:
            return np.pi
        else:
            return np.pi/2

# CHEBYSHEV POLYNOMIALS
def Ti(Ncoefs, x):
    P = np.zeros((Ncoefs, N))
    P[0] = 1
    P[1] = x
    for i in range(2, Ncoefs):
        P[i] = 2*x*P[i-1]-P[i-2] # Chebyshev polynomials
        
    return P

# DERIVATIVE OF THE POLYNOMIALS
def dTi(Ncoefs):
    dP = np.zeros((Ncoefs, Ncoefs))
    dP[1,0] = 1
    dP[2,1] = 4
    for i in range(3, Ncoefs):
        dP[i][i-1] = 2*i
        for j in range(0, i-2):
            dP[i][j] = i/(i-2)*dP[i-2][j]

    return dP

# PRODUCT OF x AND THE POLYNOMIALS
def xTi(Ncoefs):
    xP = np.zeros((Ncoefs, Ncoefs))
    xP[0,1] = 1
    xP[1,2] = xP[1,0] = 0.5
    for i in range(2, Ncoefs-1):
        xP[i][i+1] = 0.5
        xP[i][i-1] = 0.5

    xP[-1][-2] = 0.5
    return xP

# PRODUCT OF 1/x AND THE POLYNOMIALS
def Ti_x(Ncoefs):
    P_x = np.zeros((Ncoefs, Ncoefs))
    P_x[1,0] = 1
    for i in range(2, Ncoefs):
        P_x[i][i-1] = 2
        for j in range(0, i-2):
            P_x[i][j] = -P_x[i-2][j]

    return P_x

# MATRICES
P = Ti(Ncoefs, x)
dP = dTi(Ncoefs)
P_x = Ti_x(Ncoefs)

#-----------------------------------------------------------------------------

# INTERPOLATION FUNCTION
def interpolate(s, Ncoefs, P, w, N):
    # Coefficients of the interpolation
    si = np.zeros(Ncoefs)
    
    for i in range(Ncoefs):
        si[i] = CGL(s*P[i], w, N)/CGL(P[i]*P[i], w, N)
    
    return si

#-----------------------------------------------------------------------------

# CONSTRUCT THE ENTRIES OF THE MATRICES

# A_ij = a_ik * integral{ Tk * Tj }

Ld = np.zeros((Ncoefs, Ncoefs)) # First derivative
L_x = np.zeros((Ncoefs, Ncoefs)) # P/x

for i in range(Ncoefs):
    for j in range(Ncoefs):
        val1 = 0
        val2 = 0
        for k in range(Ncoefs):
            val1 += (2-np.eye(Ncoefs)[i,0])*dP[j,k]*TiTj(i,k)
            val2 += (2-np.eye(Ncoefs)[i,0])*P_x[j,k]*TiTj(i,k)
        Ld[i,j] = val1/np.pi
        L_x[i,j] = val2/np.pi
        
Ldd = np.dot(Ld,Ld)

# LINEAR OPERATOR
L = Ldd
        
#-----------------------------------------------------------------------------

# INITIAL CONFIGURATION
y = -3*np.exp(-(5-r)**2) + np.cos(2*np.pi/10*r)

# SOURCE TERM
def source(y):
    s = -y*np.sin(y)
    ds = -np.sin(y) - y*np.cos(y)
    
    return K**2*s, K**2*ds

s, ds = source(y)

# Coefficients of the interpolation
si = interpolate(s, Ncoefs, P, w, N)
ci = interpolate(y, Ncoefs, P, w, N)

s_inter = np.dot(si,P)
y_inter = np.dot(ci,P)

# BOUNDARY CONDITIONS
# u(-1) = A, u(1) = B # DIRICHLET

A = 1
B = 1
for i in range(Ncoefs):
    L[-2,i] = (-1)**i
    L[-1,i] = 1

si[-2] = A
si[-1] = B

#-----------------------------------------------------------------------------

# DEFINE THE EQUATIONS

# EQUATIONS
def eqs(Nvars, Ncoefs, ci, si):
    f = np.zeros(Nvars*Ncoefs)

    for i in range(Ncoefs):
        val = 0
        for j in range(Ncoefs):
            val += L[i,j]*ci[j]

        f[i] = val - si[i]
    
    return f

# JACOBIAN MATRIX
def Jij(Nvars, Ncoefs, ci, ds, P):
    J = np.zeros((Nvars*Ncoefs, Nvars*Ncoefs))
    
    for i in range(Ncoefs-2):
        for j in range(Ncoefs):
            val = CGL(ds*P[i]*P[j], w, N)/CGL(P[i]*P[i], w, N)
            
            J[i,j] = L[i,j] - val
    
    J[Ncoefs-2, :] = L[Ncoefs-2, :]
    J[Ncoefs-1, :] = L[Ncoefs-1, :]
    
    return J

#-----------------------------------------------------------------------------

# NEWTON-RAPHSON METHOD

from numpy.linalg import inv

for i in range(Niters):    
    f = eqs(Nvars, Ncoefs, ci, si)
    
    norm = np.dot(f,f)
    print(norm)
    if norm < 1e-5:
        break
    
    y_sol = 0
    J = Jij(Nvars, Ncoefs, ci, ds, P)
    
    Jinv = inv(J)
    X = np.dot(Jinv, f)
    for j in range(Ncoefs):
        ci[j] -= X[j]
        y_sol += ci[j]*P[j]
    
    #print(y_sol[0], y_sol[-1])
    plt.plot(x, y_sol)
    
    # REFRESH VALUES
    s, ds = source(y_sol)

    si = interpolate(s, Ncoefs, P, w, N)
    si[-2] = A
    si[-1] = B

plt.show()