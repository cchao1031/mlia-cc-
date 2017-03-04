# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from os import listdir

def create_dataset():
    X = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    y = np.array([0, 0, 1, 1])
    return X, y

#knlables(k nearest lables)
def classify0(x, X, y, k):
    distances = np.sqrt(np.sum(np.square(X - x), axis=1))
    sorted_distances_indicies = distances.argsort()
    knlables = y[sorted_distances_indicies[:k]]
    label = int(stats.mode(knlables)[0])
    return label
    
def file2matrix(path):
    data = pd.read_table(path, header = None)
    n, m = data.shape
    X = np.array(data.ix[:, :m-2])
    labels= np.array(data.ix[:, m-1])
    y = np.zeros(n)
    label_set = set(labels)
    i = 1
    label_dict = {}
    for label in label_set:
        y[labels == label] = i
        label_dict[i] = label
        i += 1
    return X, y, label_dict

def auto_norm(X):#归一化
    min_max_scaler = MinMaxScaler()
    min_max_scale_X = min_max_scaler.fit_transform(X)
    return min_max_scale_X, min_max_scaler

def dating_class_test(k):
    X, y, label_dict = file2matrix('D:/mlia(cc)/Ch02/datingTestSet.txt')
    min_max_scale_X, scaler = auto_norm(X)
    n, m = X.shape
    test_num = int(n * 0.1)
    X_train = min_max_scale_X[test_num: n]; y_train = y[test_num: n]
    error_count = 0.
    for i in range(test_num):
        classifier_result = classify0(min_max_scale_X[i], X_train, y_train, k)
        print "the classifier came back with: %s, the real answer is: %s" %(classifier_result, y[i])
        if classifier_result != y[i]:
            error_count += 1
    print "the total error rate is: %f" % (error_count/test_num)
    print error_count

def classify_person():
    X, y, label_dict = file2matrix('D:/mlia(cc)/Ch02/datingTestSet.txt')
    min_max_scale_X, scaler = auto_norm(X)
    
    ffMiles = float(raw_input("frequent flier iles earned per year?"))
    percentTats = float(raw_input("percentage of time spent playing video game?"))
    iceCream = float(raw_input("liter of ice cream consumed per year?"))
    person = np.array([ffMiles, percentTats, iceCream])
    x = scaler.transform(person)
    
    classifier_result = classify0(x, min_max_scale_X, y, 4)
    print "You will probably like this person:", label_dict[classifier_result]
    
def img2vector(path):
    return_vect = np.zeros(1024)
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
    return X, y

#hw_labels(handwriting labels)
def handwriting_class_test():
    X_train, y_train = files2mat('D:/mlia(cc)/Ch02/trainingDigits')
    X_test, y_test = files2mat('D:/mlia(cc)/Ch02/testDigits')
    test_num = X_test.shape[0]
    error_count = 0.
    for i in range(test_num):
        classifier_result = classify0(X_test[i], X_train, y_train, 3)
        print "the classifier came back with: %s, the real answer is: %s" %(classifier_result, y_test[i])
        if classifier_result != y_test[i]:
            error_count += 1
    print "the total error rate is: %f" % (error_count/test_num)
    print error_count














