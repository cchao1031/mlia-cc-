# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:09:16 2016

@author: apple
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(path):
    data = pd.read_table(path, header = None)
    n, m = data.shape
    X = np.array(data.ix[:, :m-2])
    y = np.array(data.ix[:, m-1])
    return X, y

#i: index of the alpha, m: num of alphas
def select_j_rand(i, n):#??????????
    j = i
    while j == i:
        j = np.random.randint(0, n)
    return j

def clip_alpha(alpha, H, L):
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L
    return alpha

def smo_simple(X, y, C, toler, num_iter):
    X = np.mat(X); y = np.mat(y).T
    b = 0; n, m = X.shape
    alphas = np.mat(np.zeros((n, 1)))
    iteration = 0
    while iteration < num_iter:
        alpha_paires_changed = 0
        for i in range(n):
            gx_i = np.float(np.multiply(alphas, y).T * (X * X[i, :].T)) + b
            Ei = gx_i - y[i]
            if ((y[i] * gx_i < 1 - toler) and (alphas[i] < C)) or \
                ((y[i] * gx_i > 1 + toler) and (alphas[i] > 0)):
                j = select_j_rand(i, n)
                gx_j = np.float(np.multiply(alphas, y).T * (X * X[j, :].T)) + b
                Ej = gx_j - y[j]
                alpha_i_old = alphas[i].copy(); alpha_j_old = alphas[j].copy()
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                if y[i] == y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print 'L == H'; continue
                eta =  X[i, :] * X[i, :].T + X[j, :] * X[j, :].T - 2. * X[i, :] * X[j, :].T
                if eta <= 0: print 'eta <= 0'; continue
                alphas[j] += y[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if np.abs(alphas[j] - alpha_j_old) < 1e-05: print 'j not moving enough';continue
                alphas[i] += (alpha_j_old - alphas[j]) * y[j] * y[i]
                b_i = b - Ei - y[i] * X[i,:] * X[i, :].T * (alphas[i] - alpha_i_old) -\
                    y[j] * X[i, :] * X[j, :].T * (alphas[j] - alpha_j_old)
                b_j = b - Ej - y[i] * X[i,:] * X[j, :].T * (alphas[i] - alpha_i_old) -\
                    y[j] * X[j, :] * X[j, :].T * (alphas[j] - alpha_j_old)
                if 0 < alphas[i] and alphas[i] < C: b = b_i
                elif 0 < alphas[j] and alphas[j] < C: b = b_j
                else: b = (b_i + b_j) / 2.
                alpha_paires_changed += 1
                print 'iter: %d i: %d, pairs changed %d' %(iteration, i, alpha_paires_changed)
        if alpha_paires_changed == 0: iteration += 1
        else: iteration = 0
        print 'iteration number: %d' % iteration
    return b.getA1(), alphas.getA1()

def plot_best_fit(b, alphas, X, y):
    X_mat = np.mat(X); y_mat = np.mat(y).T; alphas_mat = np.mat(alphas).T
    w = X_mat.T * np.multiply(alphas_mat, y_mat)
    w = np.r_[b, w.getA1()]
    SVs = X[alphas > 0]; y_SVs = y[alphas > 0]
    x1 = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 20)
    x2 = -(w[0] + w[1] * x1)/w[2]
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=y, s=30, alpha=0.9)
    ax.scatter(SVs[:,0], SVs[:,1], c=y_SVs, s=150, alpha=0.4)
    ax.plot(x1, x2)

class opt_struct:
    def __init__(self, X, y, C, toler, kernel_tuple):
        self.X = np.mat(X)
        self.y = np.mat(y).T
        self.C = C
        self.toler = toler
        self.n = X.shape[0]
        self.alphas = np.mat(np.zeros((self.n, 1)))
        self.b = 0
        self.E_cache = np.mat(np.zeros((self.n, 2)))
        self.K = kernel_transpose(self.X, kernel_tuple)

def calc_Ek(oS, k):
    gx_k = np.float(np.multiply(oS.alphas, oS.y).T * oS.K[:, k]) + oS.b
    Ek = gx_k - oS.y[k]
    return Ek

def select_j(i, oS, Ei):
    max_k = -1; max_delta_E = 0; Ej = 0
    oS.E_cache[i] = [1, Ei]#赋值
    valid_E_cache_index_list = np.nonzero(oS.E_cache[:, 0].A)[0]#返回了一个非零元素对应的index的数组a
    if len(valid_E_cache_index_list) > 1:
        for k in valid_E_cache_index_list:
            if k == i: 
                continue
            Ek = calc_Ek(oS, k)
            delta_E = np.abs(Ei - Ek)
            if delta_E > max_delta_E:
                max_k = k
                max_delta_E = delta_E
                Ej = Ek
        return max_k, Ej
    else:
        j = select_j_rand(i, oS.n)
        Ej = calc_Ek(oS, j)
    return j, Ej

def update_Ek(oS, k):
    Ek = calc_Ek(oS, k)
    oS.E_cache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calc_Ek(oS, i)
    if ((oS.y[i] * Ei < oS.toler) and (oS.alphas[i] < oS.C)) or \
        ((oS.y[i] * Ei >  oS.toler) and (oS.alphas[i] > 0)):
        j, Ej = select_j(i, oS, Ei)
        alpha_i_old = oS.alphas[i].copy(); alpha_j_old = oS.alphas[j].copy()
        if oS.y[i] != oS.y[j]:
            L = np.max((0, oS.alphas[j] - oS.alphas[i]))
            H = np.min((oS.C, oS.C + oS.alphas[j] - oS.alphas[i]))
        else:
            L = np.max((0, oS.alphas[j] + oS.alphas[i] - oS.C))
            H = np.min((oS.C, oS.alphas[j] + oS.alphas[i]))
        if L == H: 
            #print 'L == H'
            return 0
        eta =  oS.K[i, i] + oS.K[j, j] - 2. * oS.K[i, j]
        if eta <= 0: 
            #print 'eta <= 0'
            return 0
        oS.alphas[j] += oS.y[j] * (Ei - Ej) / eta
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        update_Ek(oS, j)
        if np.abs(oS.alphas[j] - alpha_j_old) < 1e-05:
            #print 'j not moving enough'
            return 0
        oS.alphas[i] += (alpha_j_old - oS.alphas[j]) * oS.y[j] * oS.y[i]
        update_Ek(oS, i)
        b_i = oS.b - Ei - oS.y[i] * oS.K[i, i] * (oS.alphas[i] - alpha_i_old) -\
            oS.y[j] * oS.K[i, j] * (oS.alphas[j] - alpha_j_old)
        b_j = oS.b - Ej - oS.y[i] * oS.K[i, j] * (oS.alphas[i] - alpha_i_old) -\
            oS.y[j] * oS.K[j, j] * (oS.alphas[j] - alpha_j_old)
        if 0 < oS.alphas[i] and oS.alphas[i] < oS.C: oS.b = b_i
        elif 0 < oS.alphas[j] and oS.alphas[j] < oS.C: oS.b = b_j
        else: oS.b = (b_i + b_j) / 2.
        return 1
    else: return 0

def smo_Platt(X, y, C, toler, num_iter, kernel_tuple):
    oS = opt_struct(X, y, C, toler, kernel_tuple)
    iteration = 0
    entire_set = True; alpha_pairs_changed = 0
    while (iteration < num_iter) and ((alpha_pairs_changed > 0) or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(oS.n):
                alpha_pairs_changed += innerL(i, oS)
                #print 'entire set, iter: %d i: %d, pairs changee %d' %(iteration, i, alpha_pairs_changed)
            iteration += 1
        else:
            non_SVs_index_list= np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
            for i in non_SVs_index_list:
                alpha_pairs_changed += innerL(i, oS)
                #print 'non-SVs set, iter: %d i: %d, pairs changee %d' %(iteration, i, alpha_pairs_changed)
            iteration += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print 'interation number: %d' %iteration
    return oS.b.getA1(), oS.alphas.getA1()
 
def classify_svm(b, alphas_SVs, X_SVs, y_SVs, X_input, kernel_tuple):
    X_SVs_mat = np.mat(X_SVs); y_SVs_mat = np.mat(y_SVs).T; X_input_mat = np.mat(X_input); alphas_SVs_mat = np.mat(alphas_SVs).T
    gX = kernel_transpose((X_input_mat, X_SVs_mat), kernel_tuple) *\
        np.multiply(alphas_SVs_mat, y_SVs_mat) + b
    y_predict = np.sign(gX).getA1()
    return y_predict
    
def distance_square_mat(A, B):#input A[n, k], B[m, k] output D[n, m]
    n = A.shape[0]; m = B.shape[0]
    M = A * B.T
    H1 = np.tile(np.sum(np.square(A), 1), (1, m))
    H2 = np.tile(np.sum(np.square(B), 1).T, (n, 1))
    D_square = H1 + H2 - 2 * M
    return D_square
    
def kernel_transpose(mat_or_tuple, kernel_tuple):
    if type(mat_or_tuple) is tuple:
        X_test, X_SVs = mat_or_tuple
        if kernel_tuple[0] == 'lin':
            K = X_test * X_SVs.T
        elif kernel_tuple[0] == 'rbf':
            D_square = distance_square_mat(X_test, X_SVs)
            K = np.exp(- D_square / (2 * np.square(kernel_tuple[1])))
    else:
        X = mat_or_tuple
        n, m = X.shape
        if kernel_tuple[0] == 'lin':
            K = X * X.T
        elif kernel_tuple[0] == 'rbf':
             G = X * X.T
             H = np.tile(G.diagonal(), (n, 1))
             D_square = H + H.T -2 * G
             K = np.exp(- D_square / (2 * np.square(kernel_tuple[1])))
        else:
            raise NameError('We Have a Problem -- That Kernel is not recogenized')
    return K

def test_rbf(k1 = 1.3):
    X_train, y_train = load_dataset('D:/mlia(cc)/Ch06/testSetRBF.txt')
    b, alphas = smo_Platt(X_train, y_train, 200, 0.0001, 10000, ('rbf', k1))
    n_train = len(y_train)
    num_SVs = sum(alphas > 0)
    print 'there are % d SVs' % num_SVs
  
    y_train_predict = classify_svm(b, alphas, X_train, y_train, X_train, ('rbf', k1))
    error_rate_train = sum(y_train_predict != y_train) / np.float(n_train)
    print 'the training error rate is: %f' %error_rate_train
    X_test, y_test = load_dataset('D:/mlia(cc)/Ch06/testSetRBF2.txt')
    n_test = len(y_test)
    y_test_predict = classify_svm(b, alphas, X_train, y_train, X_test, ('rbf', k1))
    error_rate_test = sum(y_test_predict != y_test) / np.float(n_test)
    print 'the test error rate is: %f' %error_rate_test
    
    SVs = X_train[alphas > 0]; y_SVs = y_train[alphas > 0]
    fig, ax = plt.subplots()
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, s=30, alpha=0.9)
    ax.scatter(SVs[:,0], SVs[:,1], c=y_SVs, s=150, alpha=0.4)
    ax.set_title('sigma = %f' %k1)

    
from os import listdir

def img2vector(path):
    return_vect = np.zeros( 1024)
    fr = open(path)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            return_vect[32*i + j] = int(lineStr[j])
    return return_vect

def files2mat(path):
    y = []
    file_list = listdir(path)
    m = len(file_list)
    X = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        y.append(class_num_str)
        X[i] = img2vector(path + '/%s' % file_name_str)
    y = np.array(y)
    X = X[(y == 1) + (y == 9)]
    y = y[(y == 1) + (y == 9)]; y[y == 9] = -1
    return X, y

def test_digits(kernel_tuple = ('rbf', 10)):
    X_train, y_train = files2mat('D:/mlia(cc)/Ch02/trainingDigits')
    b, alphas = smo_Platt(X_train, y_train, 200, 0.0001, 10000, kernel_tuple)
    n_train = len(y_train)
    num_SVs = sum(alphas > 0)
    print 'there are % d SVs' % num_SVs
    print 'sigma = %f' % kernel_tuple[1]

  
    y_train_predict = classify_svm(b, alphas, X_train, y_train, X_train, kernel_tuple)
    error_rate_train = sum(y_train_predict != y_train) / np.float(n_train)
    print 'the training error rate is: %f' %error_rate_train
    X_test, y_test = files2mat('D:/mlia(cc)/Ch02/testDigits')
    n_test = len(y_test)
    y_test_predict = classify_svm(b, alphas, X_train, y_train, X_test, kernel_tuple)
    error_rate_test = sum(y_test_predict != y_test) / np.float(n_test)
    print 'the test error rate is: %f' %error_rate_test
















    