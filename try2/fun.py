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