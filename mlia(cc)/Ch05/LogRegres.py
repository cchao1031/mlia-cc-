# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 02:10:21 2016

@author: apple
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def load_dataset(path):
    data = pd.read_table(path, header = None)
    n, m = data.shape
    constant = np.ones(n)
    X = np.array(data.ix[:, :m-2])
    y = np.array(data.ix[:, m-1])
    X = np.c_[constant, X]
    return X, y

def sigmoid(z):
    return 1./(1 + np.exp(-z))

def gradient_descent(X, y):
    X = np.mat(X); y = np.mat(y).transpose()
    n, m = X.shape
    alpha = 0.001
    max_cycles = 500
    
    w = np.mat(np.random.randn(m)).transpose()#[m, 1]
    for k in range(max_cycles):
        h = sigmoid(X * w)
        error = y - h
        w = w + alpha * X.transpose() * error
    w = w.getA1()
    return w

def plot_best_fit(w, X, y):
    x1 = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 20)
    x2 = -(w[0] + w[1]*x1)/w[2]
    fig, ax = plt.subplots()
    ax.scatter(X[:,1], X[:,2], c=y, s=100, alpha=0.7)
    ax.plot(x1, x2)

def plot_iter_w(iter_w):
    iters = len(iter_w)
    iter_w0 = [w[0] for w in iter_w]
    iter_w1 = [w[1] for w in iter_w]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(range(1,1 + iters), iter_w0)
    ax[0].set_ylabel('w0')
    ax[1].plot(range(1,1 + iters), iter_w1)
    ax[1].set_ylabel('w1')

def stoc_gradient_descent(X, y, num_iter = 150, alpha = 0.01, seed = 1031):
    n, m = X.shape
    w = np.random.randn(m)
    iter_w = []
    np.random.seed(seed)
    for k in range(num_iter):
        r = range(n)
        np.random.shuffle(r)
        for i in range(n):
            alpha_ = 4/(1. + i + k) + alpha
            h = sigmoid(np.sum(w * X[r[i]]))
            error = y[r[i]] - h
            w = w + alpha_ * X[r[i]] * error
            iter_w.append(w)
    return w#, iter_w

def classify(X, w):
    X = np.mat(X); w = np.mat(w).transpose()
    p = sigmoid(X * w)
    p = p.getA1()
    return (p > 0.5).astype(int)

def colic_test(num_iter = 500, alpha = 0.01, seed = 1031):
    X_train, y_train = load_dataset('D:/mlia(cc)/Ch05/horseColicTraining.txt')
    X_test, y_test = load_dataset('D:/mlia(cc)/Ch05/horseColicTest.txt')
    n_test = len(y_test)
    
    w = stoc_gradient_descent(X_train, y_train, num_iter, alpha, seed)
    
    y_predict = classify(X_test, w)
    error_rate = 1 - np.sum(y_predict == y_test)/float(n_test)
    print 'the error rate is : %8f' %error_rate
    return error_rate
    
def average_err(num_iter = 500, alpha = 0.01):
    err = 0.
    for i in range(10):
        err += colic_test(num_iter, alpha, seed = i)
    print 'the average error rate is : %8f' % (err/10)



    
    

















    
    