"""
@author: HuidobroMG

"""

"""
THE IDEA OF THE GALERKIN METHOD IS TO CHOOSE A BASIS OF
POLYNOMIALS THAT DIRECTLY SATISFIES THE BC
"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import Polynomials as p

# Grid parameters
N = 100 # Number of points in the grid
Npols = 8 # Number of Chebyshev polynomials
Ncoefs = Npols-2 # Number of coefficients

x = np.zeros(N)
w = np.zeros(N)

for i in range(N):
    x[i] = -np.cos(np.pi*i/(N-1)) # (CHEBYSHEV-GAUSS-LOBATTO NODES)
    w[i] = np.pi/N # (CHEBYSHEV-GAUSS-LOBATTO WEIGHTS)
w[0] /= 2
w[-1] /= 2


# Chebyshev polynomials
T = p.Ti(Npols, x)
dT = p.dTi(Npols)

# Source term
s = np.exp(x) - 4*np.exp(1)/(1+np.exp(2))

# Coefficients of the interpolation
si = p.interpolate(s, Npols, T, w, N)
s_inter = np.dot(si, T)

# Change of basis matrix
# G_2k = T_2k - T_0
# G_2k+1 = T_2k+1 - T_1

M = np.zeros((Ncoefs, Npols))
for i in range(Ncoefs):
    if i%2 == 0:
        M[i, 0] = -1
        M[i, i+2] = 1
    else:
        M[i, 1] = -1
        M[i, i+2] = 1

# Change to Galerkin basis
siG = np.zeros(Ncoefs)
for i in range(Ncoefs):
    for j in range(Npols):
        siG[i] += M[i,j]*p.TiTj(j,j)*si[j]/np.pi

# Construct the matrix of first and second derivatives of the polynomials
Ld = p.Derivative(Npols, dT)
Ldd = np.dot(Ld, Ld)

# Linear operator of the differential equation
L = Ldd - 4*Ld + 4*np.eye(Npols)

MLM = np.zeros((Ncoefs, Ncoefs))
for n in range(Ncoefs):
    for k in range(Ncoefs):
        for i in range(Npols):
            for j in range(Npols):
                MLM[n,k] += M[n,i]*L[i,j]*M[k,j]*p.TiTj(i,i)/np.pi

# Invert the matrix and obtain the solution    
Minv = inv(MLM)
yG_sol = np.dot(Minv, siG) # In Galerkin basis

# Obtain the solution in the former basis
y_sol = 0
for i in range(Npols):
    for j in range(Ncoefs):
        y_sol += M[j,i]*yG_sol[j]*T[i]

# Compare the spectral methods solutions with the actual analytical solution
real_sol = np.exp(x) - np.sinh(1)/np.sinh(2)*np.exp(2*x) - np.exp(1)/(1+np.exp(2))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, real_sol, 'b.')
ax.plot(x, y_sol, 'r-')

plt.show()