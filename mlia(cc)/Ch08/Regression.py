# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:59:02 2016

@author: apple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_dataset(path):
    data = pd.read_table(path, header = None)
    m = data.shape[1]
    X = np.array(data.ix[:,: m-2])
    y = np.array(data.ix[:, m-1])
    return X, y
    
def standard_regression(X, y):
    X_mat = np.mat(X); y_mat = np.mat(y).T
    theta_mat = (X_mat.T * X_mat).I * X_mat.T * y_mat
    return theta_mat.getA1()

def plot_fit(X, y, y_predict):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 1], y, alpha = 0.7)
    sorted_indecies = X[:, 1].argsort()
    ax.plot(X[:, 1][sorted_indecies], y_predict[sorted_indecies], 'r')
    
def distance_square_mat(A, B):#input A[n, m], B[l, m] output D[n, l]
    n = A.shape[0]; l = B.shape[0]
    M = A * B.T
    H1 = np.tile(np.sum(np.square(A), 1), (1, l))
    H2 = np.tile(np.sum(np.square(B), 1).T, (n, 1))
    D_square = H1 + H2 - 2 * M
    return D_square

def lwlr(X_train, y_train, X_input, sigma = 1.):
    #locally weighted linear regression
    l, m = X_input.shape
    X_train_mat = np.mat(X_train); X_input_mat = np.mat(X_input)
    y_train_mat = np.mat(y_train).T
    D_square = distance_square_mat(X_train_mat, X_input_mat)
    W = np.exp(- D_square / (2 * np.square(sigma)))
    all_theta_mat = np.mat(np.zeros((l, m)))
    for i in range(l):
        w_i_mat = np.mat(np.diag(W[:, i].getA1()))
        XTX = X_train_mat.T * w_i_mat * X_train_mat
        theta_mat = XTX.I * X_train_mat.T * w_i_mat * y_train_mat
        all_theta_mat[i] = theta_mat.T
    y_predict_mat = np.sum(np.multiply(all_theta_mat, X_input_mat), axis = 1)
    return y_predict_mat.getA1()

def ridge_regression(X_train, y_train, lam = 0.2):
    X_train_mat = np.mat(X_train)
    y_train_mat = np.mat(y_train).T
    n, m = X_train_mat.shape
    XTX = X_train_mat.T  * X_train_mat
    theta_mat = (XTX + lam * np.eye(m)).I * X_train_mat.T * y_train_mat
    return theta_mat.getA1()

from sklearn.preprocessing import scale
def rr_coef_plot(X, y):
    num_test = 30
    all_theta = np.zeros((num_test, X.shape[1]))
    index = []
    for i in range(num_test):
        theta = ridge_regression(scale(X), y, np.exp(i - 10))
        all_theta[i] = theta
        index.append(i - 10)
    feature_name = ['intercept', 'A','B','C','D','F','G','H']
    index = pd.Series(index, name = 'log(lambda)')
    all_theta_df = pd.DataFrame(all_theta, columns = feature_name, index = index)
    all_theta_df.plot()
  
def square_error(y_predict, y):
    return np.sum(np.square(y_predict - y)) 
    
def forward_stepwise_regreassion(X_train, y_train, yita = 0.01, num_iter = 100):
    X_train_scale_mat = np.mat(scale(X_train))
    y_train_mat = np.mat(y_train).T
    y_mean = np.mean(y_train_mat)
    y_train_mat = y_train_mat - y_mean
    n, m = X_train.shape
    all_theta = np.mat(np.zeros((num_iter, m)))
    theta_mat = np.mat(np.zeros((m,1)))
    for iteration in range(num_iter):
        print theta_mat.getA1()
        lowest_error = np.inf
        for j in range(m):
            for sign in [1., -1.]:
                theta_try_mat = theta_mat.copy()
                theta_try_mat[j] += yita * sign
                y_try_mat = X_train_scale_mat * theta_try_mat
                error = square_error(y_try_mat, y_train_mat)
                if error < lowest_error:
                    lowest_error = error
                    theta_best_mat = theta_try_mat.copy()
        theta_mat = theta_best_mat.copy()
        all_theta[iteration] = theta_mat.T
    return all_theta
    
def fsr_plot(all_theta):
    all_theta_df = pd.DataFrame(all_theta)
    all_theta_df.plot()



















    