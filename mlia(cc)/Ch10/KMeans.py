# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:41:27 2016

@author: apple
"""

import numpy as np
import pandas as pd

def load_dataset(path):
    data = pd.read_table(path, header = None)
    return np.array(data)
    
#type(data) == np.ndarray,花式索引会复制到新数组中
def random_centroids(data, k):
    n = data.shape[0]
    #np.random.seed(1031)
    random_index = np.random.permutation(n)[:k]
    return data[random_index]

def distance_square_mat(A, B):#input A[n, m], B[k, m] output D[n, k], matirx
    n = A.shape[0]; k = B.shape[0]
    M = A * B.T
    H1 = np.tile(np.sum(np.square(A), 1), (1, k))
    H2 = np.tile(np.sum(np.square(B), 1).T, (n, 1))
    D_square_mat = H1 + H2 - 2 * M
    return D_square_mat

def distance_mat2assignment_mat(distance_mat):
    n = distance_mat.shape[0]
    cluster = distance_mat.argsort(axis = 1)[:, 0]
    assignment_mat = np.c_[cluster, distance_mat[range(n), cluster.getA1()].T]
    return assignment_mat
    
def k_means(data, k):
    n = data.shape[0]
    global_cluster_mat = np.mat(np.zeros((n, 1)))
    centroids = random_centroids(data, k)
    centroids_mat = np.mat(centroids); data_mat = np.mat(data)
    cluster_changed = True
    while cluster_changed:
        D_square_mat = distance_square_mat(data_mat, centroids_mat)#[n, k]
        assignment_mat = distance_mat2assignment_mat(D_square_mat)#[n, 2]
        if (assignment_mat[:, 0] == global_cluster_mat).all():
            cluster_changed = False
        global_cluster_mat = assignment_mat[:, 0].copy()
        #print centroids_mat
        for c in range(k):
            points_in_cluster = data[np.nonzero(assignment_mat[:, 0].getA1() == c)[0]]
            centroids_mat[c, :] = np.mean(points_in_cluster, axis = 0)
    return centroids_mat, assignment_mat

def bisecting_k_means(data, k):
    n = data.shape[0]
    centroids = np.mean(data, axis = 0)
    centroids_mat = np.mat(centroids); data_mat = np.mat(data)
    assignment_mat = np.c_[np.mat(np.zeros(n)).T, distance_square_mat(data_mat, centroids_mat)]
    while centroids_mat.shape[0] < k:
        lowest_sse = np.inf
        for c in range(centroids_mat.shape[0]):
            for i in range(3):
                points_in_cluster = data[np.nonzero(assignment_mat[:, 0].getA1() == c)[0]]
                cluster_centroids_mat, cluster_assignment_mat = k_means(points_in_cluster, 2)
                sse_split = np.sum(cluster_assignment_mat[:, 1])
                sse_no_split = \
                    np.sum(assignment_mat[np.nonzero(assignment_mat[:, 0].getA1() != c)[0]])
                print 'sse_split:%0.2f sse_no_split:%0.2f' %(sse_split, sse_no_split)
                if (sse_split + sse_no_split) < lowest_sse:
                    best_cluster2split = c
                    best_new_centroids_mat = cluster_centroids_mat.copy()
                    best_new_assignment_mat = cluster_assignment_mat.copy()
                    lowest_sse = sse_split + sse_no_split
        best_new_assignment_mat[np.nonzero(best_new_assignment_mat[:, 0].getA1() == 0)[0], 0] = \
                                best_cluster2split
        best_new_assignment_mat[np.nonzero(best_new_assignment_mat[:, 0].getA1() == 1)[0], 0] = \
                                centroids_mat.shape[0]
        print 'now we have %d clusters' %(centroids_mat.shape[0] + 1)
        centroids_mat[best_cluster2split] = best_new_centroids_mat[0]
        centroids_mat = np.r_[centroids_mat, best_new_centroids_mat[1]]
        assignment_mat[np.nonzero(assignment_mat[:, 0].getA1() == best_cluster2split)[0]] = \
                        best_new_assignment_mat
    return centroids_mat, assignment_mat
    

    
    
    
    
    
    
    

