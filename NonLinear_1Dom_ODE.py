"""
@author: HuidobroMG

"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import Polynomials as p

# Grid parameters
N = 200 # Number of points in the grid
Ncoefs = 10 # Number of coefficients in the interpolation

# Create the grid
x = np.zeros(N) # Space points
w = np.zeros(N) # Weights
# Chebyshev-Gauss-Lobatto nodes and weights
for i in range(N):
    x[i] = -np.cos(np.pi*i/(N-1))
    w[i] = np.pi/N
w[0] /= 2
w[-1] /= 2

# Compactification of the space coordinate
# r = C/2*(1+x), r = [0, C]
C = 4
r = C/2*(1+x)
alpha = C/2

# Compute the Chebyshev polynomials
T = p.Ti(Ncoefs, x)
dT = p.dTi(Ncoefs)
T_x = p.Ti_x(Ncoefs)

# Construct the matrices of coefficients
Ld = p.Derivative(Ncoefs, dT)
L_x = p.Cocient(Ncoefs, T_x)        
Ldd = np.dot(Ld, Ld)

# Linear operator of the differential equation
L = Ldd + alpha*Ld

# The non-linearity of the differential equation requires an initial configuration
y = -0.25*r + 1

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y, 'r-')

# Source term
def source(y):
    s = -y**2
    ds = -2*y
    return alpha**2*s, alpha**2*ds

s, ds = source(y)

# Coefficients of the interpolation
si = p.interpolate(s, Ncoefs, T, w, N)
ci = p.interpolate(y, Ncoefs, T, w, N)

s_inter = np.dot(si, T)
y_sol = np.dot(ci, T)

# Dirichlet boundary conditions, u(-1) = A, u(1) = B
A = 1
B = 0
for i in range(Ncoefs):
    L[-2,i] = (-1)**i
    L[-1,i] = 1

si[-2] = A
si[-1] = B

# Residuals of the differential equation
def residuals(ci, si):
    f = np.dot(L, ci) - si
    return f

# Define the Jacobian matrix
def Jij(Ncoefs, ci, ds, T):
    J = np.zeros((Ncoefs, Ncoefs))
    for i in range(Ncoefs-2):
        for j in range(Ncoefs):
            val = p.CGL(ds*T[i]*T[j], w, N)/p.CGL(T[i]*T[i], w, N)
            J[i,j] = L[i,j] - val
    
    J[Ncoefs-2, :] = L[Ncoefs-2, :]
    J[Ncoefs-1, :] = L[Ncoefs-1, :]
    return J

# Newton-Raphson iterative method
tol = 1e-10
Niters = 10
for i in range(Niters):
    f = residuals(Ncoefs, ci, si)
    J = Jij(Ncoefs, ci, ds, T)

    # Check the residual error
    res = np.dot(f, f)
    print('|f|**2 = ', res)
    if res < tol:
        ax.plot(x, y_sol, 'g-')
        break

    Jinv = inv(J)
    X = np.dot(Jinv, f)
    ci -= X
    y_sol = np.dot(ci, T)

    ax.plot(x, y_sol, 'b-')
    
    # Refresh the value
    s, ds = source(y_sol)
    si = p.interpolate(s, Ncoefs, T, w, N)
    si[-2] = A
    si[-1] = B

plt.show()