"""
@author: HuidobroMG

"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import Polynomials as p

# Grid parameters
N = 100 # Number of points in the grid
Ncoefs = 8 # Number of coefficients in the interpolation

# Create the grid
x = np.zeros(N) # Space points
w = np.zeros(N) # Weights
# Chebyshev-Gauss-Lobatto nodes and weights
for i in range(N):
    x[i] = -np.cos(np.pi*i/(N-1))
    w[i] = np.pi/N
w[0] /= 2
w[-1] /= 2

# Compute the Chebyshev polynomials
T = p.Ti(Ncoefs, x)
dT = p.dTi(Ncoefs)

# Source term
s = np.exp(x) - 4*np.exp(1)/(1+np.exp(2))

# Coefficients of the interpolated source
si = p.interpolate(s, Ncoefs, T, w, N)
s_interp = np.dot(si, T)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(x, s, 'b.')
ax1.plot(x, s_interp, 'r-')

ax1.set_xlabel('x', fontsize = 15)
ax1.set_ylabel('s(x)', fontsize = 15)

# Construct the matrix of first and second derivatives of the polynomials
Ld = p.Derivative(Ncoefs, dT)
Ldd = np.dot(Ld, Ld)

# Linear operator of the differential equation
L = Ldd - 4*Ld + 4*np.eye(Ncoefs)

# We adopt the Tau method

# Dirichlet boundary conditions, u(-1) = u(1) = 0
for j in range(Ncoefs):
    L[Ncoefs-2,j] = (-1)**j
    L[Ncoefs-1, j] = 1

# Erase the corresponding equations
si[Ncoefs-1:] = 0
si[Ncoefs-2:] = 0

# Invert the matrix and obtain the solution
Linv = inv(L)

sol = np.dot(Linv, si)
y_sol = np.dot(sol, T)

# Compare the spectral methods solutions with the actual analytical solution
real_sol = np.exp(x) - np.sinh(1)/np.sinh(2)*np.exp(2*x) - np.exp(1)/(1+np.exp(2))

ax2 = fig.add_subplot(122)
ax2.plot(x, real_sol, 'b.', label = 'Numerical solution')
ax2.plot(x, y_sol, 'r-', label = 'Analytical solution')

ax2.set_xlabel('x', fontsize = 15)
ax2.set_ylabel('y(x)', fontsize = 15)

ax2.legend(fontsize = 12)

plt.show()