#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:30:29 2018

@author: paolograniero
"""
#%%
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from fun import *
from sklearn.preprocessing import scale

h_gamma = 0.0001
seed = 1733715
np.random.seed(seed = seed)

dataset = pd.read_csv('2017.12.11 Dataset Project 2.csv', header = 0)

data = pd.DataFrame()

for col_name, column in dataset.iteritems():
    data[col_name] = scale(dataset[col_name])
data.y = dataset.y


X = data.iloc[:, :-1].values
Y = (data.iloc[:, -1].values*2 - 1).reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = seed)

L = len(X_train)

#%%
#grid
score_list = dict()
gamma_values = [2**-e for e in range(0,20)]
C_values = [2**-e for e in range(-5,5)]
tot_params = len(gamma_values)*len(C_values)
j = 1
for gamma in gamma_values:
    for C in C_values:

        Q = Q_matrix(X_train, Y_train, gamma)

        const = lambda lam: con(lam, Y_train)
        constraint = {'type': 'eq', 'fun': const}

        bound = tuple((0,C) for i in range(L))

        res = minimize(fun = obj,
                       x0 = np.zeros(shape=(L,1)),
                       args = (Q),
                       method='SLSQP',
                       bounds = bound,
                       constraints= constraint)


        lam_star = res.x
        for i in range(L):
            if lam_star[i] > 10**-6: break
        b = b_star(lam_star, X_train, Y_train, gamma, i)

        if res.success:
            score = np.mean(np.multiply(np.transpose(Y_test),pred(X_test, lam_star, b, X_train, Y_train, gamma))[0] < 0)
            score_list[(gamma, C)] = score

        print('\rRemaining parameters: %3d' %(tot_params-j), end = '')
        j += 1
#%%
min_score = 1
opt_par = 0
for par, sc in score_list.items():
    if sc < min_score:
        min_score = sc
        opt_par = par
print('Optimal parameters: %s\nTest accuracy: %f' %(opt_par, 1-min_score))
#%%
gamma, C = (0.125, 2) # optimal parameters

Q = Q_matrix(X_train, Y_train, gamma)

const = lambda lam: con(lam, Y_train)
constraint = {'type': 'eq', 'fun': const}

bound = tuple((0,C) for i in range(L))

res = minimize(fun = obj,
               x0 = np.zeros(shape=L),
               args = (Q),
               method='SLSQP',
               bounds = bound,
               constraints= constraint)


lam_star = res.x
for i in range(L):
    if lam_star[i] > 10**-6: break
b = b_star(lam_star, X_train, Y_train, gamma, i)
train_accuracy = 1 - np.mean(np.multiply(np.transpose(Y_train),pred(X_train, lam_star, b, X_train, Y_train, gamma))[0] < 0)
test_accuracy = 1 - np.mean(np.multiply(np.transpose(Y_test),pred(X_test, lam_star, b, X_train, Y_train, gamma))[0] < 0)
print('Train accuracy: %f\nTest Accuracy: %f' %(train_accuracy, test_accuracy))