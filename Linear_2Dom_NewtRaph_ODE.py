# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:38:17 2021

@author: huido
"""

import numpy as np
import matplotlib.pyplot as plt

Nvars = 1 # Number of variables
Ncoefs = 5 # Number of coefficients in each domain
Niters = 50 # Number of iterations
N = 100 # Number of points in the grid

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

#-----------------------------------------------------------------------------

# DOMAINS: r = [-1, 1], r1 = [-1, 0], r2 = [0, 1]

# DOMAIN 1: r1 = [-1, 0], x1 = [-1, 1]
# r1 = (x1 - 1)/2

x1 = x
alpha1 = 0.5 # dr/dx
r1 = (x1-1)/2

# DOMAIN 2: r2 = [0, 1], x2 = [-1, 1]
# r2 = (x2 + 1)/2

x2 = x
alpha2 = 0.5 # dr/dx
r2 = (x2+1)/2

r = np.concatenate((r1, r2))

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
def Ti(Ncoefs, x, N):
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

# CHEBYSHEV POLYNOMIALS AND MATRICES OF COEFFICIENTS
P = Ti(Ncoefs, x, N)
dP = dTi(Ncoefs)

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

for i in range(Ncoefs):
    for j in range(Ncoefs):
        val = 0
        for k in range(Ncoefs):
            val += (2-np.eye(Ncoefs)[i,0])*dP[j,k]*TiTj(i,k)
        Ld[i,j] = val/np.pi

# Second derivative
Ldd = np.dot(Ld,Ld)

#-----------------------------------------------------------------------------

# INITIAL CONFIGURATION
y1 = np.exp(-r1**2)*np.cos(1)*np.exp(1)
y2 = np.exp(-r2**2)*np.cos(1)*np.exp(1)

# SOURCE TERM
def source(y, alpha):
    
    s = -y
    ds = -1
    
    return alpha**2*s, alpha**2*ds

s1, ds1 = source(y1, alpha1)
s2, ds2 = source(y2, alpha2)

# COEFFICIENTS OF THE INITIAL CONFIGURATION AND SOURCE

ci_1 = interpolate(y1, Ncoefs, P, w, N)
ci_2 = interpolate(y2, Ncoefs, P, w, N)
si_1 = interpolate(s1, Ncoefs, P, w, N)
si_2 = interpolate(s2, Ncoefs, P, w, N)

#-----------------------------------------------------------------------------

# LINEAR OPERATORS

# DOMAIN 1

# LINEAR OPERATOR
L1 = (Ldd+Ldd)/2

# DOMAIN 2

# LINEAR OPERATOR
L2 = (Ldd+Ldd)/2

# CONCATENATE THE POLYNOMIALS
Pt = [P, P]

#-----------------------------------------------------------------------------

# BOUNDARY CONDITIONS
# y(-1) = A, y(1) = B
A = np.cos(1)
B = np.cos(1)

# DOMAIN 1
# y1(x = -1) = A
for i in range(Ncoefs):
    L1[-2, i] = (-1)**i

si_1[-2] = A

# DOMAIN 2
# y2(x = 1) = B
L2[-2,:] = 1
si_2[-2] = B


# CONCATENATE BOTH DOMAINS
L = np.zeros((2*Ncoefs, 2*Ncoefs))
L[:Ncoefs, :Ncoefs] = L1
L[Ncoefs:, Ncoefs:] = L2

si = np.concatenate((si_1, si_2))
ds = [ds1, ds2]
ci = np.concatenate((ci_1, ci_2))


# IMPOSE CONTINUITY ON THE FIELD AND THE DERIVATIVE

for i in range(Ncoefs):
    # phi1(x = 1) = phi2(x = -1)
    L[Ncoefs-1, i] = -1
    L[Ncoefs-1, i+Ncoefs] = (-1)**i
    
    # dphi1/dx(x = 1) = dphi2/dx(x = -1)
    L[2*Ncoefs-1, i] = sum(Ld[:, i])
    L[2*Ncoefs-1, i+Ncoefs] = (-1)**i*sum(Ld[:, i])

si[Ncoefs-1] = 0
si[2*Ncoefs-1] = 0

#-----------------------------------------------------------------------------

# DEFINE THE EQUATIONS

# EQUATIONS
def eqs(Nvars, Ncoefs, ci, si):
    f = np.zeros(2*Ncoefs)

    for i in range(2*Ncoefs):
        val = 0
        for j in range(2*Ncoefs):
            val += L[i,j]*ci[j]

        f[i] = val - si[i]
    
    return f


# JACOBIAN MATRIX
def Jij(Nvars, Ncoefs, ci, ds, P):
    J = np.zeros((2*Ncoefs, 2*Ncoefs))
    
    # DO NOT TOUCH THE BC AND CONTINUITY ROWS
    
    # FIRST QUARTER
    for i in range(Ncoefs-2):
        for j in range(Ncoefs):
            val = CGL(ds[0]*P[0][i]*P[0][j], w, N)/CGL(P[0][i]*P[0][i], w, N)
    
            J[i,j] = L[i,j] - val
    
    #SECOND QUARTER
    for i in range(Ncoefs, 2*Ncoefs-2):
        idx = i-Ncoefs
        for j in range(Ncoefs, 2*Ncoefs):
            jdx = j-Ncoefs
            val = CGL(ds[1]*P[1][idx]*P[1][jdx], w, N)/CGL(P[1][idx]*P[1][idx], w, N)
    
            J[i,j] = L[i,j] - val
    
    J[Ncoefs-2, :] = L[Ncoefs-2, :]
    J[Ncoefs-1, :] = L[Ncoefs-1, :]
    J[2*Ncoefs-2, :] = L[2*Ncoefs-2, :]
    J[2*Ncoefs-1, :] = L[2*Ncoefs-1, :]    
    
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
    
    J = Jij(Nvars, Ncoefs, ci, ds, Pt)
    Jinv = inv(J)
    X = np.dot(Jinv, f)
    
    for j in range(2*Ncoefs):
        ci[j] -= X[j]
    
    y_sol1 = np.dot(ci[:Ncoefs], P)
    y_sol2 = np.dot(ci[Ncoefs:], P)
    
    print(y_sol1[0], y_sol2[-1])
    plt.plot(r1, y_sol1)
    plt.plot(r2, y_sol2)
    
    # REFRESH VALUES
    s1, ds1 = source(y_sol1, alpha1)
    s2, ds2 = source(y_sol2, alpha2)
    
    ds[0] = ds1
    ds[1] = ds2

    si_1 = interpolate(s1, Ncoefs, P, w, N)
    si_2 = interpolate(s2, Ncoefs, P, w, N)
    si_1[-2] = A
    si_2[-2] = B
    
    si = np.concatenate((si_1, si_2))
    si[Ncoefs-1] = 0
    si[2*Ncoefs-1] = 0

y_sol = np.concatenate((y_sol1, y_sol2))