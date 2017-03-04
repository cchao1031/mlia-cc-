# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 15:46:08 2016

@author: apple
"""
import numpy as np
import numpy.linalg as la

def loadExData():
    return np.array([[4, 4, 0, 2, 2],
                     [4, 0, 0, 3, 3],
                     [4, 0, 0, 1, 1],
                     [1, 1, 1, 2, 0],
                     [2, 2, 2, 0, 0],
                     [5, 5, 5, 0, 0],
                     [1, 1, 1, 0, 0]])

def loadExData2():
    return np.array([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])
           
'''similarity (a and b are vectors)mat'''
def euclid_sim(a, b):
    return 1. / (1. + la.norm(a - b))

def pearson_sim(a, b):
    if len(a) < 3:
        return 1.
    return 0.5 + 0.5 * np.corrcoef(a, b, rowvar = 0)[0][1]

def cos_sim(a, b):
    return 0.5 + 0.5 * (np.float(a.T * b) / (la.norm(a) * la.norm(b)))
    
similarity_method_dict = {'euclid': euclid_sim, 'pearson': pearson_sim, 'cos': cos_sim}
'''similarity (a and b are vectors)'''

#estimate scores based on similarity
def standard_estimate(data_mat, user_id, similarity_method, item_id):
    m = data_mat.shape[1]
    total_sim = 0.; total_weighted_sim = 0.
    for j in range(m):
        user_rating = data_mat[user_id, j]
        if user_rating == 0:
            continue
        overlap = np.nonzero(np.logical_and(data_mat[:, item_id].A > 0, \
                                            data_mat[:, j].A > 0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = similarity_method(data_mat[overlap, item_id], \
                                            data_mat[overlap, j])
        print 'the %d and %d similarity is %f' %(item_id, j, similarity)
        total_sim += similarity
        total_weighted_sim += similarity * user_rating
    if total_sim == 0:
        return 0
    else:
        return total_weighted_sim / total_sim

#过程简介
#为了估计使用者i的对第j个物品的评分，首先找出使用者已作出评分的物品
#计算需要估计评分的物品与已作出评分物品的相似度
#以相似度作为权重对评分进行加权平均
#结果即为使用者i的对第j个物品的评分的估计
def svd_reduce_estimate(data_mat, user_id, similarity_method, item_id):
    m = data_mat.shape[1]
    total_sim = 0.; total_weighted_sim = 0.
    U, Sigma, V_T = la.svd(data_mat)
    energy = np.square(Sigma)
    k = np.nonzero(np.cumsum(energy / np.sum(energy)) > 0.9)[0][1]
    Sigma_mat = np.mat(np.eye(k) * Sigma[:k])
    reduced_data_mat = Sigma_mat.T * U[:,:k].T * data_mat
    for j in range(m):
        user_rating = data_mat[user_id, j]
        if user_rating == 0:
            continue
        similarity = similarity_method(reduced_data_mat[:, item_id], \
                                        reduced_data_mat[:, j])
        print 'the %d and %d similarity is %f' %(item_id, j, similarity)
        total_sim += similarity
        total_weighted_sim += similarity * user_rating
    if total_sim == 0:
        return 0
    else:
        return total_weighted_sim / total_sim

import operator
def recommend(data, user_id, N = 3, similarity_method = 'cos', est_method = svd_reduce_estimate):
    similarity_method = similarity_method_dict[similarity_method]
    data_mat = np.mat(data)
    unrated_items_id = np.nonzero(data_mat[user_id, :].A == 0)[1]
    if len(unrated_items_id) == 0:
        return 'you rated everything'
    item_scores = []
    for item_id in unrated_items_id:
        est_score  = est_method(data_mat, user_id, similarity_method, item_id)
        item_scores.append((item_id, est_score))
    return sorted(item_scores, key = operator.itemgetter(1), reverse = True)[:N]
    
def  print_mat(in_mat, thresh = 0.8):
    for i in range(32):
        for j in range(32):
            if float(in_mat[i, j]) > thresh:
                print 1,
            else:
                print 0,
        print ''

def txt2mat(path):
    m = []
    for line in open(path).readlines():
        new_row = []
        for i in range(32):
            new_row.append(int(line[i]))
        m.append(new_row)
    return np.mat(m)

def img_compress(path, thresh = 0.8):
    m = txt2mat(path)
    print '****original matrix****'
    print_mat(m, thresh)
    
    U, Sigma, V_T = la.svd(m)
    energy = np.square(Sigma)
    k = np.nonzero(np.cumsum(energy / np.sum(energy)) > 0.9)[0][1]
    Sigma_mat = np.mat(np.eye(k) * Sigma[:k])
    reconstruct_mat = U[:, :k] * Sigma_mat * V_T[:k, :]
    print '****reconstructed matrix with %.2f%% information****' %(np.cumsum(energy / np.sum(energy))[k-1]*100)
    print_mat(reconstruct_mat, thresh)















