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

# D1: r1 = [-1, 0], x = [-1, 1]
# r1 = (1 - x)/2
r1 = (x-1)/2
alpha1 = 0.5 # dx/dx1

# D2: r2 = [0, 1], x = [-1, 1]
# r2 = (1 + x)/2
r2 = (1+x)/2
alpha2 = 0.5

r = np.concatenate((r1, r2))

# Compute the Chebyshev polynomials
T = p.Ti(Ncoefs, x)
dT = p.dTi(Ncoefs)

# Source term in the first domain
s1 = alpha1**2
si_1 = p.interpolate(s1, Ncoefs, T, w, N)

# Source term in the second domain
s2 = 0
si_2 = p.interpolate(s2, Ncoefs, T, w, N)

si = np.concatenate((si_1, si_2))

# Construct the matrix of first and second derivatives of the polynomials
Ld = p.Derivative(Ncoefs, dT)
Ldd = np.dot(Ld, Ld)

# Linear operator of the differential equation in each domain
L1 = -Ldd + alpha1**2*4*np.eye(Ncoefs)
L2 = -Ldd + alpha2**2*4*np.eye(Ncoefs)

# Concatenate both domains
L = np.zeros((2*Ncoefs, 2*Ncoefs)) # Add 2 more rows for the junction conditions
L[:Ncoefs, :Ncoefs] = L1
L[Ncoefs:, Ncoefs:] = L2

# Impose the Dirichlet boundary condition on each domain
for j in range(Ncoefs):
    L[Ncoefs-2,j] = (-1)**j # u1(-1) = 0
    L[-2, Ncoefs+j] = 1 # u2(1) = 0

# Impose continuity in the change of domains
for j in range(Ncoefs):
    # u1(x1 = 1) = u2(x2 = -1)
    L[Ncoefs-1, j] = -1
    L[Ncoefs-1, Ncoefs+j] = (-1)**j
    
    # du1(x1 = 1) = du2(x2 = -1)
    L[-1, j] = sum(Ld[:, j])
    L[-1, Ncoefs+j] = (-1)**j*sum(Ld[:, j])
    
# The structure of the matrix Lt is:
# The first Ncoefs-2 equations are the first domain eqs
# The next 3 equations are: BC on D1, continuity on the field
# The next eqs are the second domain eqs
# The last 2 equations are the BC on D2 and the continuity on the derivative

# Invert the matrix and obtain the solution
Linv = inv(L)
sol = np.dot(Linv, si)

y_sol1 = np.dot(sol[:Ncoefs], T)
y_sol2 = np.dot(sol[Ncoefs:], T)    
y_sol = np.concatenate((y_sol1, y_sol2))

# Construct the actual analytical solution and compare
B1 = -1/(8*(1+np.exp(2))) - np.exp(2)/(8*(1+np.exp(4)))
B2 = np.exp(4)/8*(np.exp(2)/(1+np.exp(4))-1/(1+np.exp(2)))
real_sol1 = 0.25-(np.exp(2)/4+B1*np.exp(4))*np.exp(2*r1) + B1*np.exp(-2*r1)
real_sol2 = B2*(np.exp(-2*r2)-np.exp(2*r2-4))

real_sol = np.concatenate((real_sol1, real_sol2))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(r1, real_sol1, 'b.')
ax.plot(r2, real_sol2, 'g.')
ax.plot(r1, y_sol1, 'r-')
ax.plot(r2, y_sol2, 'k-')

ax.set_xlabel('r', fontsize = 15)
ax.set_ylabel('y(r)', fontsize = 15)

plt.show()