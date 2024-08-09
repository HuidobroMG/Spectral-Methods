"""
@author: HuidobroMG

"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scsp
from numpy.linalg import inv
import Polynomials as p
import warnings
warnings.filterwarnings("ignore")

# Grid parameters
NcoefsL = 9
NcoefsC = 10
Nr = 100 # Number of points in the grid in r
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

# Domain in r = [1, inf], x = [-1, 1]
r = 2/(1-xr)

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
Ld = p.Derivative(NcoefsC, dT) # First derivative
Lx = p.Product(NcoefsC, xT) # x*P
Ldd = np.dot(Ld, Ld) # Second derivative
Lx_I = Lx - np.eye(NcoefsC) # x*P - I
Lx_I2 = np.dot(Lx_I, Lx_I) # (x*P - I)**2

# Initial configuration
phi = np.zeros((Nr, Ntheta))
for i in range(Nr):
    for j in range(Ntheta):
        phi[i,j] = np.cos(4*theta[j])/r[i]

# Source term, s(r, theta)
def source(phi):
    N1, N2 = np.shape(phi)
    s = 10*phi
    ds_phi = 10*np.ones((N1, N2))
    return s, ds_phi

s, ds_phi = source(phi)

# Change of basis from Chebyshev to Legendre, P_l = M_lm * T_m
M = p.T_2_P(NcoefsL, Lp, P, wtheta, Ntheta)
Minv = inv(M)

# Interpolation of a function for any set of polynomials on 2 dimensions
def interpolate_2D(f, NcoefsC, NcoefsL, T, P, wr, wtheta, Nr, Ntheta):
    # Obtain the Chebyshev interpolation coefficients on the theta grid
    ci_lT = np.zeros((NcoefsL, Nr))
    for i in range(Nr):
        ci_lT[:, i] = p.interpolate(f[i], NcoefsL, P, wtheta, Ntheta)

    ci_l = np.dot(np.transpose(Minv), ci_lT) # Coefficient curves on Legendre basis, ci_l(r)

    # Extract the matrix of coefficients
    ci = np.zeros((NcoefsL, NcoefsC))
    for i in range(NcoefsL):
        ci[i] = p.interpolate(ci_l[i], NcoefsC, T, wr, Nr)
    return ci, ci_l

ci, ci_l = interpolate_2D(phi, NcoefsC, NcoefsL, T, P, wr, wtheta, Nr, Ntheta)
si, si_l = interpolate_2D(s, NcoefsC, NcoefsL, T, P, wr, wtheta, Nr, Ntheta)


# Dirichlet boundary conditions, phi(r = 1) = cos(4*theta), phi(r = inf) = 0
g = np.cos(4*theta)
gi = p.interpolate(g, NcoefsL, P, wtheta, Ntheta)
gi_l = np.dot(np.transpose(Minv), gi)

# Residuals of the differential equation
def residuals(ci, si, L):
    f = np.dot(L, ci) - si
    return f

# Delta matrix
def Delta_lm_ij(NcoefsL, NcoefsC, ds):
    Delta = np.zeros((NcoefsL, NcoefsL, NcoefsC, NcoefsC))
    
    val = np.zeros(Nr)
    for l in range(NcoefsL):
        for m in range(l+1):
            for i in range(NcoefsC):
                for j in range(i+1):
                    for k in range(Nr):
                        val[k] = p.CGL(ds[i]*P[l]*P[m], wtheta, Ntheta)/p.CGL(P[l]*P[l], wtheta, Ntheta)
                    Delta[l,m,i,j] = p.CGL(val*T[i]*T[j], wr, Nr)/p.CGL(T[i]*T[i], wr, Nr)
                    Delta[l,m,j,i] = Delta[l,m,i,j]
                    Delta[m,l,i,j] = Delta[l,m,i,j]
    
    return Delta

# Jacobian matrix
def Jij(NcoefsL, NcoefsC, Delta, L, l):
    J = np.zeros((NcoefsL, NcoefsC, NcoefsC))
    
    Ds = np.zeros((NcoefsC, NcoefsC))
    for m in range(NcoefsL):
        Ds[:] = 0
        for n in range(NcoefsL):
            for o in range(NcoefsL):
                Ds += Minv[n,l]*M[m,o]*Delta[o,n]
        if l == m:
            J[m] = L - Ds
        else:
            J[m] = -Ds
    
    # The boundary conditions rows
    for m in range(NcoefsL):
        if m == l:
            J[m,-2,:] = L[-2,:]
            J[m,-1,:] = L[-1,:]
    
    return J


# Newton-Raphson iterative method
Niters = 20
tol = 1e-10

f = np.zeros((NcoefsL, NcoefsC))
J = np.zeros((NcoefsL, NcoefsL, NcoefsC, NcoefsC))

fc = np.zeros((NcoefsL*NcoefsC))
cic = np.zeros((NcoefsL*NcoefsC))
Jc = np.zeros((NcoefsL*NcoefsC, NcoefsL*NcoefsC))
for iteration in range(Niters):
    print('iteration = ', iteration)
    res_max = 0
    res_val = 0
    
    Delta = Delta_lm_ij(NcoefsL, NcoefsC, ds_phi)
    for l in range(NcoefsL):
        # Linear operator
        L = np.dot(Lx_I2, Ldd) - l*(l+1)*np.eye(NcoefsC)
        
        # Boundary conditions, phi_l(xr = -1) = gi_l, phi(xr = 1) = 0
        for j in range(NcoefsC):
            L[-2,j] = (-1)**j
        L[-1,:] = 1
        si[l,-2] = gi_l[l]
        si[l,-1] = 0
        
        # Compute the residuals
        f[l] = residuals(ci[l], si[l], L)
        res_val = np.dot(f[l], f[l])
        if res_val > res_max:
            res_max = res_val
        
        # Compute the jacobian matrix
        J[l] = Jij(NcoefsL, NcoefsC, Delta, L, l)
    
    print('|f|**2 = ', res_max)
    if res_max < tol:
        break
    
    # Convert the tensors into compactified matrices to invert
    for l in range(NcoefsL):
        for i in range(NcoefsC):
            cic[int(i+NcoefsC*l)] = ci[l,i]
            fc[int(i+NcoefsC*l)] = f[l,i]
            for m in range(NcoefsL):
                for j in range(NcoefsC):
                    Jc[int(i+NcoefsC*l),int(j+NcoefsC*m)] = J[l,m,i,j]
    Jcinv = inv(Jc)
    Xc = np.dot(Jcinv, fc)
    cic -= Xc
    
    # Recover the matrix of coefficients
    for l in range(NcoefsL):
        for i in range(NcoefsC):
            ci[l,i] = cic[i+NcoefsC*l]
    
    # Refresh the value
    y_sol = np.dot(np.transpose(np.dot(ci, T)), Lp)
    s, ds_phi = source(y_sol)
    si, si_l = interpolate_2D(s, NcoefsC, NcoefsL, T, P, wr, wtheta, Nr, Ntheta)


idx = np.where(r > 5)[0][0]
r = r[:idx]
y_sol = y_sol[:idx, :]

R, THETA = np.meshgrid(r, theta, indexing = 'ij')
y, z = R*np.sin(THETA), R*np.cos(THETA)

plt.contourf(y, z, y_sol)
plt.colorbar()

plt.show()
