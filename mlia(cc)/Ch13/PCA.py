# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 08:44:19 2016

@author: apple
"""

import pandas as pd
import numpy as np

def load_dataset(path):
    data = pd.read_table(path, header = None)
    return np.array(data)

def pca(data, top_N_feature = 9999999):
    feature_mean = np.mean(data, axis = 0)
    mean_removed_data = data - feature_mean
    covariance_mat = np.mat(np.cov(mean_removed_data, rowvar = 0))
    eig_values, eig_vectors = np.linalg.eig(covariance_mat)
    eig_values_sorted_index = np.argsort(eig_values)
    index = eig_values_sorted_index[::-1][: top_N_feature]
    top_N_eig_vectors = eig_vectors[:, index]
    low_dimension_mat = np.mat(mean_removed_data) * top_N_eig_vectors
    reconstruct_data = (low_dimension_mat * top_N_eig_vectors.T) + feature_mean
    return low_dimension_mat.getA(), reconstruct_data.getA()
    