#! /usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from functionsGiac import *

# Importing dataset
dataset = pd.read_csv('2017.12.11 Dataset Project 2.csv', header = 0, sep = ',')
X = dataset.iloc[:, :-1].values
y = 2*(dataset.iloc[:, -1].values.reshape(-1,1)) - 1 # Mapping {0, 1} to {-1, 1}

# Split train, validation, test set
X_train_n_validation, X_test, y_train_n_validation, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1733715)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_n_validation, y_train_n_validation, test_size = 0.3, random_state = 1733715)
# Q matrix of the dual problem
Q = Q_matrix(X_train, y_train, 1)
# Inintialize vector of Lagrage multipliers
L = X_train.shape[0]
alpha = np.zeros((L, 1))
print(objective(alpha, Q))
# initial guesses
n = len(X_train)
x0 = np.zeros((n,))
# show initial objective
print('Initial Objective: ' + str(objective(x0, Q)))

'''
GRIDSEARCH C,GAM
'''
# optimize
C= 2
b = (0, C)
bnds = tuple(b for i in range(n))
con = lambda alpha: constraint1(y_train, alpha)
con1 = {'type': 'eq', 'fun': con}
cons = ([con1])
solution = minimize(objective, x0, args = (Q), method='SLSQP', bounds=bnds, constraints=cons)
x = solution.x

# show final objective
print('Final Objective: ' + str(objective(x, Q)))

# print solution
print('Solution')

gamma_values = [2**e for e in range(-2,3,1)]
C_values = [2**e for e in range(-2,3,1)]

score = dict()
L = X_train.shape[0]
x0 = np.zeros((L,))
for gamma in gamma_values:
    for C in C_values:
        score[(C, gamma)] = dict()
        Q = Q_matrix(X_train, y_train, gamma)
        b = (0, C)
        bnds = tuple(b for i in range(L))
        con = lambda alpha: constraint1(y_train, alpha)
        con1 = {'type': 'eq', 'fun': con}
        cons = ([con1])
        solution = minimize(objective, x0, args = (Q), method='SLSQP', bounds=bnds, constraints=cons)
        x = solution.x
        print('Final Objective: ' + str(objective(x, Q)))
