from numpy import diag, tril, triu, dot, ones, zeros
from numpy.linalg import norm
from scipy.linalg import solve_triangular

def is_sdd(A):
    n = len(A)
    sdd = True
    for i in range(n):
        sumR=sum(abs(A[i,:]))-abs(A[i,i])
        if abs(A[i,i])-sumR <= 0:
            sdd=False
            break
    return sdd

def decompose(A):
    D = diag(diag(A))
    L = tril(A) - D
    U = triu(A) - D
    return D, L, U

def jacobi_step(D, L, U, b, xk):
    n = len(b)
    S = D
    T = L + U
    rhs = b - dot(T,xk)
    xkp1 = zeros(n)
    for i in range(n):
        xkp1[i] = rhs[i] / D[i,i]
    return xkp1
 
def jacobi_iteration(A, b, x0, epsilon=1e-8):
    D, L, U = decompose(A)
    xk = x0 + 1
    xkp1 = x0
    while (norm(xkp1 - xk,2) > epsilon):
        xk = xkp1
        xkp1 = jacobi_step(D, L, U, b, xk)
    return xkp1

def gauss_seidel_step(D, L, U, b, xk):
    n = len(b)
    S = D + U
    T = L
    rhs = b - dot(T,xk)
    xkp1 = solve_triangular(S, rhs)
    return xkp1

def gauss_seidel_iteration(A, b, x0, epsilon=1e-8):
    D, L, U = decompose(A)
    xk = x0 + 1
    xkp1 = x0
    while (norm(xkp1 - xk,2) > epsilon):
        xk = xkp1
        xkp1 = gauss_seidel_step(D, L, U, b, xk)
    return xkp1
