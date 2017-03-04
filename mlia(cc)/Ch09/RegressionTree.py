# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:23:49 2016

@author: apple
"""

class decision_node:
    def __init__(self, feature=None, value=None, result=None, T_branch=None, F_branch=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.T_branch = T_branch
        self.F_branch = F_branch

import pandas as pd
import numpy as np

#data 的格式为dataframe
def load_data(path, names = None):#data is a DataFrame
    data = pd.read_table(path, header = None, names = names)
    return data

def divide_set(data, feature, value):
    if isinstance(value, np.int64) or isinstance(value, np.float64):
        judge = data.ix[:, feature] >= value
    else:
        judge = data.ix[:, feature] == value

    set_T = data[judge]
    set_F = data[-judge]
    return set_T, set_F

def choose_best_split(data, leaf_err, option):
    #可更改current_score求法
    current_score = leaf_err(data)
    
    best_gain = 0.0; best_feature = None; best_value = None; best_sets = None
    
    num_feature = len(data.columns) - 1
    for feature in range(num_feature):
        feature_values = data.ix[:, feature].unique()
        if len(feature_values) == 1:
            continue
        else:
            for value in feature_values:
                set_T, set_F = divide_set(data, feature, value)
                if (set_T.shape[0] > option) and (set_F.shape[0] > option):
                    #可更改new_score求法
                    new_score = leaf_err(set_T) + leaf_err(set_F)
                    gain = current_score - new_score
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_value = value
                        best_sets = (set_T, set_F)
    return best_gain, best_feature, best_value, best_sets

'''leaf_type'''
def constant_mode(data):
    return np.mean(data.ix[:, -1])       
def constant_mode_err(data):
    return np.var(data.ix[:, -1]) * len(data.ix[:, -1])
def constant_mode_predict(tree, single_data):
    return tree.result

def linear_solve(data):   
    n, m = data.shape
    data_mat = np.mat(data)
    X_mat = np.mat(np.ones((n, m)))
    X_mat[:, 1:m] = data.ix[:, :-1]; y_mat = data_mat[:, m-1]
    if np.linalg.det(X_mat.T * X_mat) == 0.:
        raise NameError('This matrix is singular, cannot do inverse')
    theta_mat = (X_mat.T * X_mat).I * X_mat.T * y_mat
    return theta_mat, X_mat, y_mat
def linear_mode(data):
    theta_mat, X_mat, y_mat = linear_solve(data)
    return theta_mat.getA1()
def linear_mode_err(data):
    theta_mat, X_mat, y_mat = linear_solve(data)
    y_predict_mat = X_mat * theta_mat
    return np.sum(np.square(y_mat - y_predict_mat))
def linear_mode_predict(tree, single_data):
    theta_mat_transposed = np.mat(tree.result)
    x_mat = np.c_[1., np.mat(single_data)].T
    return float(theta_mat_transposed * x_mat)

leaf_type = {}
leaf_type['constant'] = (constant_mode, constant_mode_err, constant_mode_predict)
leaf_type['linear'] = (linear_mode, linear_mode_err, linear_mode_predict)
'''leaf type'''
    
def build_tree(data, mode = 'constant', option = 10):
    global leaf_type
    leaf_mode, leaf_err, _ = leaf_type[mode]
    best_gain, best_feature, best_value, best_sets = choose_best_split(data, leaf_err, option)
    if best_gain > 0:
        T_branch = build_tree(best_sets[0], mode, option)
        F_branch = build_tree(best_sets[1], mode, option)
        return decision_node(feature=data.columns[best_feature], value=best_value,
                             T_branch=T_branch, F_branch = F_branch)
    else:
        return decision_node(result=leaf_mode(data))

def print_tree(tree, indent=''):
    #判断是否是节点
    if type(tree.result) == np.ndarray or type(tree.result) == float:#是叶节点
        print str(tree.result)
    else:#不是叶节点
        #打印条件
        print str(tree.feature) + ':' + str(tree.value) + '?'

        #打印分支
        print indent + 'T->',
        print_tree(tree.T_branch, indent + '\t')
        print indent + 'F->',
        print_tree(tree.F_branch, indent + '\t')

def tree2diction(tree):
    tree_dict = {}
    if type(tree.result) == np.ndarray or type(tree.result) == float:#是叶节点
        return tree.result
    else:
        tree_dict['feature'] = tree.feature
        tree_dict['value'] = tree.value
        tree_dict['T_branch'] = tree2diction(tree.T_branch)
        tree_dict['F_branch'] = tree2diction(tree.F_branch)
    return tree_dict

def is_tree(obj):
    return (type(obj.result).__name__ == 'NoneType')

def get_mean(tree):
    mean_T = None; mean_F = None
    if is_tree(tree.T_branch):
        mean_T = get_mean(tree.T_branch)
    else:
        mean_T = tree.T_branch.result
    if is_tree(tree.F_branch):
        mean_F = get_mean(tree.F_branch)
    else:
        mean_F = tree.F_branch.result
    return (mean_T + mean_F) / 2.

#tree进入这个prune函数本身就会发生变化
def prune(tree, test_data):
    if test_data.shape[0] == 0: 
        tree.result = get_mean(tree)
        return tree
    
    if (is_tree(tree.T_branch)) or (is_tree(tree.F_branch)):
        test_set_T, test_set_F = divide_set(test_data, tree.feature, tree.value)
    if is_tree(tree.T_branch):
        tree.T_branch = prune(tree.T_branch, test_set_T)
    if is_tree(tree.F_branch):
        tree.F_branch = prune(tree.F_branch, test_set_F)
    
    if not is_tree(tree.T_branch) and not is_tree(tree.F_branch):
        test_set_T, test_set_F = divide_set(test_data, tree.feature, tree.value)
        error_no_merge = np.sum(np.square(test_set_T.ix[:, -1] - tree.T_branch.result))\
            + np.sum(np.square(test_set_F.ix[:, -1] - tree.F_branch.result))
        error_merge = np.sum(np.square(test_data.ix[:, -1] - get_mean(tree)))
        if error_merge < error_no_merge:
            print 'merging'
            tree.result = get_mean(tree)
            return tree
        else:
            return tree
    else:
        return tree

def tree_predict(tree, single_data, mode = 'constant'):
    global leaf_type
    leaf_model_predict = leaf_type[mode][2]
    if single_data[tree.feature] >= tree.value:
        if is_tree(tree.T_branch):
            return tree_predict(tree.T_branch, single_data, mode)
        else:
            return leaf_model_predict(tree.T_branch, single_data)
    else:
        if is_tree(tree.F_branch):
            return tree_predict(tree.F_branch, single_data, mode)
        else:
            return leaf_model_predict(tree.F_branch, single_data)

def creat_predict(tree, data_input, mode = 'constant'):
    n_input = len(data_input)
    y_predict = []
    for i in range(n_input):
        y_predict.append(tree_predict(tree, data_input.ix[i], mode))
    return np.array(y_predict)
    
#y_predict和y都是ndarray
def R_square(y_predict, y):
    return 1. - (np.sum(np.square(y_predict - y)) / np.sum(np.square(y - np.mean(y))))
    
    
    
    
    
    
    
    
    
    
    
    
    