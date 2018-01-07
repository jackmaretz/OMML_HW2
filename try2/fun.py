#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:28:05 2018

@author: paolograniero
"""
import numpy as np

def kernel(x_i, x_j, h_gamma):
    return np.exp(-1 * h_gamma * np.linalg.norm(x_i-x_j)**2)

def Q_matrix(X, Y, h_gamma = 1):
    L = len(X)
    Q = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            Q[i,j] = Y[i]*Y[j]*kernel(X[i], X[j], h_gamma)
    return Q

def obj(lam, Q):
    return 1/2 * np.matmul(np.matmul(np.transpose(lam), Q), lam) - sum(lam)

def con(lam, Y_train):
    return np.matmul(np.transpose(lam), Y_train)

def b_star(lam, X, Y, gamma, idx):
    summation = 0
    for i in range(len(X)):
        summation += lam[i]*Y[i]*kernel(X[idx], X[i], gamma)
    return Y[idx] - summation

def pred(X_pred, lam, b, X_train, Y_train, gamma):
    L = len(lam)
    predictions = np.zeros(len(X_pred))
    idx = 0
    for x in X_pred:
        summation = 0
        for j in range(L):
            summation += lam[j]*Y_train[j]*kernel(X_train[j], x, gamma)
        predictions[idx] = np.sign(summation + b)
        idx += 1
    return predictions

def RS_set(alpha, C, y_train):
    Uplus = set()
    Uminus = set()
    Lplus = set()
    Lminus = set()
    mid = set()
    i = 0
    for a, y in zip(alpha, y_train):
        if a < 10**-6:
            if y < 0:
                Lminus.add(i)
            elif y > 0:
                Lplus.add(i)
        elif abs(C-a) < 10**-6:
            if y < 0:
                Uminus.add(i)
            elif y > 0:
                Uplus.add(i)
        if a > 10**-6 and abs(C-a) > 10**-6:
            mid.add(i)
        i += 1
    R = Lplus.union(Uminus.union(mid))
    S = Lminus.union(Uplus.union(mid))
    return list(R), list(S)

def working_set(q, grad, y, R, S):
# =============================================================================
#     I = set()
#     J = set()
#     I.update([np.argmax(vec_R)])
#     J.update([np.argmin(vec_S)])
# =============================================================================
     if (q+1)%2 == 0:
         print('Invalid q value')
         return
     q_star = int(q/2)
     vec_R = -np.multiply(y, grad)[R]
     vec_S = -np.multiply(y,grad)[S]

     I = list(vec_R.argsort())[-q_star:]
     J = list(vec_S.argsort())[:q_star]

     W = I.extend(J)

     return W


