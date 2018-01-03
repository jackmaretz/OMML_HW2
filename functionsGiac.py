import numpy as np
from scipy.optimize import minimize

def gauss_kernel(x_i, x_j, gamma):
    '''RBF kernel function, gamma hyperparameter'''
    return np.exp(-1*gamma*np.linalg.norm(x_i-x_j)**2)

def poly_kernel(x_i, x_j, p):
    '''Polynomial kernel functionn, p hyperparameter'''
    if p < 1:
        print('Invalid value for parameter p')
        return
    return (np.dot(x_i, x_j) + 1)**p

def Q_matrix(X, y, gamma):
    L = X.shape[0]
    Q_temp = np.ones((L,L))

    for i in range(L):
        for j in range(i, L):
            Q_temp[i, j] = y[i] * y[j] * gauss_kernel(X[i], X[j], gamma)
    Q = 0.5 * (Q_temp + Q_temp.T)
    return Q 

def objective(alpha, Q):
    L = Q.shape[0]
    
    q = 0.5*np.dot(np.dot(alpha.T, Q), alpha)
    
    e = np.ones((L,1))
    s = np.dot(e.T, alpha)
    
    return (q - s)


def constraint1(y,alpha):
    return np.dot(y.T,alpha)


def getBeta():
    f = 1


def fHyperPlane(X, sol_X, Y, b_star, gamma):
    L = X.shape[0]
    K = np.ones((L, L))
    for i in range(L):
        for j in range(i, L):
            K[i, j] = gauss_kernel(X[i], X[j], gamma)
    return np.sign(sum(np.dot(np.dot(sol_X, Y), K)) + b_star)
