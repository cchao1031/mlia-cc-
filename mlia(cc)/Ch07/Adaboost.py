# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:04:16 2016

@author: apple
"""
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt

def load_simpel_data():
    X = np.array([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    return X, y

def load_dataset(path):
    data = pd.read_table(path, header = None)
    n, m = data.shape
    X = np.array(data.ix[:, :m-2])
    y = np.array(data.ix[:, m-1])
    return X, y
    
def stump_classify(X_j, value, y):#已经并非传统意义的stump了，有可能全部都为正或全为负，关键是为了降低错误率
    n = len(y); y_predict = np.zeros(n)
    if np.dtype(value) == 'float64':
        set_T = (X_j >= value)
    else:
        set_T = (X_j == value)
        print '------------------------------------'
    set_T_label = mode(y[set_T])[0][0]
    y_predict[set_T] = set_T_label
    if sum(set_T) < n:
        set_F_label = mode(y[-set_T])[0][0]
        y_predict[-set_T] = set_F_label
        return y_predict, set_T_label, set_F_label
    return y_predict, set_T_label, 0.
    

def build_stump(X, y, D):
    #X, y, D都是数组, X元素要么是np.float, 要么是字符
    n, m = X.shape
    best_Stump = {}
    min_error = np.inf
    for j in range(m):
        feture_j_values = np.unique(X[:, j])
        for value in feture_j_values:
            y_predict, set_T_label, set_F_label = stump_classify(X[:, j], value, y)
            error_array = np.ones(n)
            error_array[y_predict == y] = 0
            weighted_error = np.sum(D * error_array)
            #print 'split: dim %d, threshold %s, the weighted error is %.2f' % (j, value, weighted_error)
            if weighted_error < min_error:
                min_error = weighted_error
                best_Stump['dim'] = j
                best_Stump['threshold'] = value
                best_Stump['set_T_label'] = set_T_label
                best_Stump['set_F_label'] = set_F_label
                best_y_predict = y_predict.copy()
    return best_Stump, min_error, best_y_predict

def AdaBoost_train_DS(X, y, num_iter = 50):
    week_classifiers = []; week_classifiers_list = []
    n = len(y)
    D = np.ones(n)/n
    sum_weighted_y_predict = np.zeros(n)
    for i in range(num_iter):
        best_Stump, error, y_predict = build_stump(X, y, D)
        #print 'D:', D
        alpha = 0.5 * np.log((1 - error) / np.max(error, 1e-16))
        best_Stump['alpha'] = alpha
        week_classifiers.append(best_Stump)
        #print len(week_classifiers)
        if i in [0, 9, 49, 99, 499, 999]:
            week_classifiers_list.append(list(week_classifiers))
        #print 'y predict:', y_predict
        D = D * np.exp(-alpha * y * y_predict)
        D = D / np.sum(D)
        sum_weighted_y_predict += alpha * y_predict
        #print 'sum_weighted_y_predict:', np.sign(sum_weighted_y_predict)
        total_error_rate = np.sum(1. * (np.sign(sum_weighted_y_predict) != y)) / n
        print total_error_rate
        #print 'total error rate:', total_error_rate,'\n'
        if total_error_rate == 0: 
            break
    return week_classifiers_list, sum_weighted_y_predict
    
def classify_adaboost(week_classifiers, X):
    X_mat = np.mat(X)
    n = X_mat.shape[0]
    T = len(week_classifiers)
    alphas = [classify['alpha'] for classify in week_classifiers]
    dims = np.array([classify['dim'] for classify in week_classifiers])
    thresholds = [classify['threshold'] for classify in week_classifiers]
    set_T_labels = [classify['set_T_label'] for classify in week_classifiers]
    set_F_labels = [classify['set_F_label'] for classify in week_classifiers]
    sum_weighted_y_predict = np.zeros(n)
    for t in range(T):
        y_predict = np.zeros(n)
        if np.dtype(thresholds[t]) == 'float64':
            set_T = X_mat[:, dims[t]] >= thresholds[t]
        else:
            set_T = X_mat[:, dims[t]] == thresholds[t]
        y_predict[set_T.getA1()] = set_T_labels[t]
        y_predict[y_predict == 0] = set_F_labels[t]
        sum_weighted_y_predict += alphas[t] * y_predict
        #print 'sum_weighted_y_predict:', sum_weighted_y_predict
    return np.sign(sum_weighted_y_predict)

def test_Adaboost():
    X_train, y_train = load_dataset('D:/mlia(cc)/Ch05/horseColicTraining.txt')
    X_test, y_test = load_dataset('D:/mlia(cc)/Ch05/horseColicTest.txt')
    y_train[y_train == 0] = -1; y_test[y_test == 0] = -1
    n_train = len(y_train); n_test = len(y_test)
    week_classifiers_list = AdaBoost_train_DS(X_train, y_train, 1000)
    for i in range(len(week_classifiers_list)):
        error_count_train = np.sum(classify_adaboost(week_classifiers_list[i], X_train) != y_train)
        error_rate_train = error_count_train / np.float(n_train)
        error_count_test = np.sum(classify_adaboost(week_classifiers_list[i], X_test) != y_test)
        error_rate_test = error_count_test / np.float(n_test)
        print '%4d week classifiers, trainning error rate %.3f, test error rate %.3f' \
            %(len(week_classifiers_list[i]), error_rate_train, error_rate_test)

def plot_ROC(predict_strength, y):
    PR_list = [[0., 0.]]#positive rate list, contain[FPR,TPR]
    FPR = 0.; TPR = 0.
    TPR_sum = 0
    num_positive_class = np.sum(y == 1)
    FPR_step = 1. / (len(y) - num_positive_class)
    TPR_step = 1. / num_positive_class
    sorted_indecies = np.argsort(-predict_strength)
    for index in sorted_indecies:
        if y[index] == 1:
            TPR += TPR_step
        else:
            FPR += FPR_step
            TPR_sum += TPR
        PR_list.append(list((FPR, TPR)))
    PR_list.append([1., 1.])
    PR_array = np.array(PR_list)
    fig, ax = plt.subplots()
    ax.plot(PR_array[:, 0], PR_array[:, 1], c = 'b')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True positive Rate')
    ax.set_title('ROC curve for Adaboost Horse Colic Detection System')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); 
    plt.show()
    print 'the Area Under ROC Curve is', TPR_sum * FPR_step
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    