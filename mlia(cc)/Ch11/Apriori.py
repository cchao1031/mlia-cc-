# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 00:46:04 2016

@author: apple
"""

def load_dataset():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#C means candidate
def create_C_1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)

def scan_dataset(dataset, C_k, min_support):
    count_set = {}
    for transaction in dataset:
        for candidate in C_k:
            if candidate.issubset(transaction):
                if not count_set.has_key(candidate):
                    count_set[candidate] = 1
                else:
                    count_set[candidate] += 1
    n = float(len(dataset))
    return_list = []
    support_data = {}
    for comb in count_set:
        support = count_set[comb] / n
        if support >= min_support:
            return_list.append(comb)
        support_data[comb] = support
    return return_list, support_data
        
def create_C_k_p_1(C_k, L_k, k):
    return_set = set([])
    len_L_k = len(L_k)
    unfreq_set = set(C_k) - set(L_k)
    print unfreq_set
    for i in range(len_L_k):
        for j in range(i + 1, len_L_k):
            L1 = list(L_k[i])[: k - 1]; L2 = list(L_k[j])[: k -1]
            L1.sort(), L2.sort()
            if L1 == L2:
                if len(L_k[0]) == 1:
                    return_set.update([L_k[i] | L_k[j]])
                else:
                    if unfreq_set == set():
                        return_set.update([L_k[i] | L_k[j]])
                    else:
                        for unfreq in unfreq_set:
                            if not unfreq.issubset(L_k[i] | L_k[j]):
                                return_set.update([L_k[i] | L_k[j]])
    return list(return_set)

def apriori(dataset, min_support = 0.5):
    C_1 = create_C_1(dataset)
    L_1, support_data = scan_dataset(dataset, C_1, min_support)
    L = [L_1]; C_k = C_1
    k = 1
    while len(L[k-1]) > 0:
        C_k_p_1 = create_C_k_p_1(C_k, L[k-1], k)
        C_k = C_k_p_1
        L_k_p_1, support_data_k_p_1 = scan_dataset(dataset, C_k_p_1, min_support)
        support_data.update(support_data_k_p_1)
        L.append(L_k_p_1)
        k += 1
        print k 
    L.remove([])
    return L, support_data
        
def generate_rules(L, support_data, min_confidence = 0.7):
    rule_list = []
    for i in range(1, len(L)):
        for frequent_set in L[i]:
            H1 = [frozenset([item]) for item in frequent_set]
            if i > 1:
                rules_from_back_element(frequent_set, H1, support_data, rule_list, min_confidence)
            else:
                calculate_confidence(frequent_set, H1, support_data, rule_list, min_confidence)
    return rule_list

def calculate_confidence(frequent_set, H, support_data, rule_list, min_confidence):
    pruned_H = []
    for back_element in H:
        confidence = support_data[frequent_set] / support_data[frequent_set - back_element]
        if confidence >= min_confidence:
            print frequent_set - back_element, '--->', back_element, 'confidence:',confidence
            rule_list.append((frequent_set - back_element, back_element, confidence))
            pruned_H.append(back_element)
    return pruned_H
    
def rules_from_back_element(frequent_set, H, support_data, rule_list, min_confidence):
    m = len(H[0])#后件的大小
    if len(frequent_set) > m + 1:
        H_m_p_1 = new_create_C_k_p_1(H, m)
        H_m_p_1 = calculate_confidence(frequent_set, H_m_p_1, support_data, rule_list, min_confidence)
        if len(H_m_p_1) > 1:
            rules_from_back_element(frequent_set, H_m_p_1, support_data, rule_list, min_confidence)
    
def new_create_C_k_p_1(L_k, k):
    return_list = []
    len_L_k = len(L_k)
    for i in range(len_L_k):
        for j in range(i + 1, len_L_k):
            L1 = list(L_k[i])[: k - 1]; L2 = list(L_k[j])[: k -1]
            L1.sort(), L2.sort()
            if L1 == L2:
                return_list.append(L_k[i] | L_k[j])
    return return_list

    
    
    
    
    
    
    