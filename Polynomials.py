"""
@author: HuidobroMG

We define the mathematical background of the Chebyshev and Legendre set of polynomials required for
the resolution of differential equations using spectral methods.
"""

# Import the modules
import numpy as np
import scipy.special as scsp

# Chebyshev-Gauss-Lobatto quadrature integration
def CGL(f, w, N):
    return sum(f[i]*w[i] for i in range(N))

# Integral of Chebyshev polynomials <Ti|Tj>
def TiTj(i, j):
    if i != j:
        return 0
    else:
        if i == 0:
            return np.pi
        else:
            return np.pi/2

# Generator of Chebyshev polynomials, T(x)
def Ti(Ncoefs, x):
    N = len(x)

    T = np.zeros((Ncoefs, N))
    T[0] = 1
    T[1] = x
    for i in range(2, Ncoefs):
        T[i] = 2*x*T[i-1]-T[i-2] # Chebyshev polynomials recurrence relation
    return T

# Derivative matrix of Chebyshev polynomials
def dTi(Ncoefs):
    dT = np.zeros((Ncoefs, Ncoefs))
    dT[1,0] = 1
    dT[2,1] = 4
    for i in range(3, Ncoefs):
        dT[i][i-1] = 2*i
        for j in range(0, i-2):
            dT[i][j] = i/(i-2)*dT[i-2][j]
    return dT

# Matrix of the product x * T(x)
def xTi(Ncoefs):
    xT = np.zeros((Ncoefs, Ncoefs))
    xT[0,1] = 1
    xT[1,2] = xT[1,0] = 0.5
    for i in range(2, Ncoefs-1):
        xT[i][i+1] = 0.5
        xT[i][i-1] = 0.5

    xT[-1][-2] = 0.5
    return xT

# Matrix of the product 1/x * T(x)
def Ti_x(Ncoefs):
    T_x = np.zeros((Ncoefs, Ncoefs))
    T_x[1,0] = 1
    for i in range(2, Ncoefs):
        T_x[i][i-1] = 2
        for j in range(0, i-2):
            T_x[i][j] = -T_x[i-2][j]
    return T_x

# Coefficients matrices of the operations
def Derivative(Ncoefs, dT): # dT(x)
    Ld = np.zeros((Ncoefs, Ncoefs))
    for i in range(Ncoefs):
        for j in range(Ncoefs):
            val = 0
            for k in range(Ncoefs):
                val += (2-np.eye(Ncoefs)[i,0])*dT[j,k]*TiTj(i,k)
                
            Ld[i,j] = val/np.pi
    return Ld

def Product(Ncoefs, xT): # x * T(x)
    Lx = np.zeros((Ncoefs, Ncoefs))
    for i in range(Ncoefs):
        for j in range(Ncoefs):
            val = 0
            for k in range(Ncoefs):
                val += (2-np.eye(Ncoefs)[i,0])*xT[j,k]*TiTj(i,k)
                
            Lx[i,j] = val/np.pi
    return Lx

def Cocient(Ncoefs, T_x): # 1/x * T(x)
    L_x = np.zeros((Ncoefs, Ncoefs))
    for i in range(Ncoefs):
        for j in range(Ncoefs):
            val = 0
            for k in range(Ncoefs):
                val += (2-np.eye(Ncoefs)[i,0])*T_x[j,k]*TiTj(i,k)

            L_x[i,j] = val/np.pi
    return L_x

# Legendre polynomials generator, P(x)
def Legendre(Ncoefs, x):
    N = len(x)
    P = np.zeros((Ncoefs, N))
    for i in range(Ncoefs):
        P[i] = scsp.eval_legendre(i, x)
    return P

# Change of basis Chebyshev to Legendre
def T_2_P(Ncoefs, P, T, w, N):
    M = np.zeros((Ncoefs, Ncoefs))
    for i in range(Ncoefs):
        for j in range(Ncoefs):
            M[i,j] = CGL(P[i]*T[j], w, N)/CGL(T[j]*T[j], w, N)
    return M

# Interpolation of a function for any set of polynomials
def interpolate(f, Ncoefs, T, w, N):
    # Coefficients of the interpolation
    fi = np.zeros(Ncoefs)
    for i in range(Ncoefs):
        fi[i] = CGL(f*T[i], w, N)/CGL(T[i]*T[i], w, N)
    return fi