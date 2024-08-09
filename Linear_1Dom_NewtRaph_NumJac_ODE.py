"""
@author: HuidobroMG

"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import Polynomials as p

# Grid parameters
Ncoefs = 10
N = 150

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
T_x = p.Ti_x(Ncoefs)

# Construct the matrices of coefficients
Ld = p.Derivative(Ncoefs, dT) # First derivative
L_x = p.Cocient(Ncoefs, T_x) # T/x 
Ldd = np.dot(Ld, Ld) # Second derivative

# The non-linearity of the differential equation requires an initial configuration
y = np.exp(-x**2)*np.cos(1)/np.exp(-1)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y, 'r-', label = 'Initial configuration')

# Source term and its derivative
def source(y):
    s = -y
    ds = -1
    return s, ds

s, ds = source(y)

# Coefficients of the interpolation
si = p.interpolate(s, Ncoefs, T, w, N)
ci = p.interpolate(y, Ncoefs, T, w, N)

s_inter = np.dot(si, T)
y_sol = np.dot(ci, T)

# Dirichlet boundary conditions, u(-1) = u(1) = cos(1)
for i in range(Ncoefs):
    Ldd[-2,i] = (-1)**i
    Ldd[-1,i] = 1

si[-2] = np.cos(1)
si[-1] = np.cos(1)

# Residuals of the differential equation
def residuals(Ncoefs, ci, si):
    f = np.zeros(Ncoefs)
    for i in range(Ncoefs):
        val = 0
        for j in range(Ncoefs):
            val += Ldd[i,j]*ci[j]
        f[i] = val - si[i]
    return f

# Jacobian matrix, J_ij = df_i/dc_j = (f_i[c_j + eps*c_j] - f_i[c_j])/(2*eps)
def Jij(Ncoefs, ci, T):
    J = np.zeros((Ncoefs, Ncoefs))
    ci_new = 1.0*ci
    F1 = np.zeros((Ncoefs, Ncoefs))
    F2 = np.zeros((Ncoefs, Ncoefs))
    for j in range(Ncoefs):
        ci_new[j] = 1.01*ci[j]
        F1[:,j] = residuals(Ncoefs, ci_new, si)
        ci_new[j] = 0.99*ci[j]
        F2[:,j] = residuals(Ncoefs, ci_new, si)
        ci_new[j] = ci[j]
    
    for i in range(Ncoefs-2):
        for j in range(Ncoefs):
            eps = 0.01*ci[j]
            J[i,j] = (F1[i,j]-F2[i,j])/(2*eps)
    
    # Boundary conditions rows
    J[Ncoefs-2, :] = Ldd[Ncoefs-2, :]
    J[Ncoefs-1, :] = Ldd[Ncoefs-1, :]
    return J


# Newton-Raphson iterative method
tol = 1e-10
Niters = 20
for i in range(Niters):
    f = residuals(Ncoefs, ci, si)
    J = Jij(Ncoefs, ci, T)

    # Check the residual error
    res = np.dot(f, f)
    print('|f|**2 = ', res)
    if res < tol:
        ax.plot(x, y_sol, 'g-', label = 'Numerical solution')
        break
    
    Jinv = inv(J)
    X = np.dot(Jinv, f)
    ci -= X
    y_sol = np.dot(ci, T)

    ax.plot(x, y_sol, 'b-')
    
    # Refresh the value
    s, ds = source(y_sol)
    si = p.interpolate(s, Ncoefs, T, w, N)
    si[-2] = np.cos(1)
    si[-1] = np.cos(1)

ax.plot(x, np.cos(x), 'k--', label = 'Analytical solution')

ax.legend(fontsize = 12)
plt.show()