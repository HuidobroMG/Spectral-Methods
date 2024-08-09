"""
@author: HuidobroMG

"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scsp
from numpy.linalg import inv
import Polynomials as p

# Grid parameters
NcoefsC = 10 # Number of Chebyshev coefficients
Nr = 100 # Number of points in the grid in r
NcoefsL = 8 # Number of Legendre coefficients
Ntheta = 50 # Number of points in the grid in theta

# Create the grid in r
xr = np.zeros(Nr)
wr = np.zeros(Nr)
# Chebyshev-Gauss-Lobatto nodes and weights
for i in range(Nr):
    xr[i] = -np.cos(np.pi*i/(Nr-1))
    wr[i] = np.pi/Nr
wr[0] /= 2
wr[-1] /= 2

# The domain in r = [1, inf], x = [-1, 1]
r = 2*(2+xr)/(1-xr)

# Chebyshev polynomials
T = p.Ti(NcoefsC, xr)
dT = p.dTi(NcoefsC)
xT = p.xTi(NcoefsC)
T_x = p.Ti_x(NcoefsC)

# Create the grid in theta
xtheta = np.zeros(Ntheta)
wtheta = np.zeros(Ntheta)
# Chebyshev-Gauss-Lobatto nodes and weights
for i in range(Ntheta):
    xtheta[i] = -np.cos(np.pi*i/(Ntheta-1))
    wtheta[i] = np.pi/Ntheta
wtheta[0] /= 2
wtheta[-1] /= 2

# The domain in theta = [0, 2*pi], x = [-1, 1]
theta = np.arccos(-xtheta)

# Legendre polynomials
P = p.Ti(NcoefsL, xtheta)
Lp = p.Legendre(NcoefsL, xtheta)

# Construct the operators
Ld = np.zeros((NcoefsC, NcoefsC)) # dT/dx
L_x = np.zeros((NcoefsC, NcoefsC)) # T/x
Ldd = np.dot(Ld, Ld) # ddT/ddx
L_xtheta = np.dot(L_x, L_x) # P/x

# Source term, s(r, theta)
s = np.zeros((Nr, Ntheta))
for i in range(Nr):
    for j in range(Ntheta):
        s[i,j] = np.cos(theta[j])/r[i]

# Obtain the Chebyshev interpolation coefficients on the theta grid
si_lT = np.zeros((NcoefsL, Nr))
for i in range(Nr):
    si_lT[:, i] = p.interpolate(s[i], NcoefsL, P, wtheta, Ntheta)

# Change of basis from Chebyshev to Legendre
M = p.T_2_P(NcoefsL, Lp, P, wtheta, Ntheta)
Minv = inv(M)
si_l = np.dot(np.transpose(Minv), si_lT) # Coefficient curves on Legendre basis, si_l(r)

# Extract the matrix of coefficients
si = np.zeros((NcoefsL, NcoefsC))
for i in range(NcoefsL):
    si[i] = p.interpolate(si_l[i], NcoefsC, T, wr, Nr)

# We may reconstruct the source curve
#y = np.dot(np.transpose(np.dot(si, T)), Lp)

# Imporse the boundary condition at r = 1
g = np.cos(theta)**2
gi = p.interpolate(g, NcoefsL, P, wtheta, Ntheta)
gi_l = np.dot(Minv, gi)

# Construct the linear part of the differential equation
for i in range(NcoefsL):
    L = Ldd + 2*np.dot(L_x, Ld) - i*(i+1)*L_xtheta
    
    
    
    












