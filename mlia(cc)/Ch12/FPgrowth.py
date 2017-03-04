# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 02:58:32 2016

@author: apple
"""

class tree_node:
    def __init__(self, name, count, parent_node):
        self.name = name
        self.count = count
        self.node_link = None
        self.parent = parent_node
        self.children = {}
        
    def increase(self, count):
        self.count += count

    def display(self, ind = 0):
        print '  '*ind, self.name, ':', self.count
        for child in self.children.values():
            child.display(ind + 1)

import operator
def creat_FP_tree(data_count_dict, min_support = 1):
    #第一次扫描dataset，创建头指针表
    header_table = {}#头指针表
    for transaction in data_count_dict:
        for item in transaction:
            header_table[item] = header_table.get(item, 0)  + data_count_dict[transaction]
    
    #删去不满足最小支持频度的项
    for k in header_table.keys():
        if header_table[k] < min_support:
            del(header_table[k])
    
    frequent_item_set = set(header_table.keys())
    
    #如果没有项满足最小支持频度则退出函数
    if len(frequent_item_set) == 0:
        return None, None
    
    for k in header_table:
        header_table[k] = [header_table[k], None]
    
    #创建空集树根
    return_tree = tree_node('Null Set', 1, None)
    
    #第二次扫描dataset
    for transaction, count in data_count_dict.items():
        local_dataset = {}
        for item in transaction:
            #提取出满足最小支持频度的项
            if item in frequent_item_set:
                local_dataset[item] = header_table[item][0]
        if len(local_dataset) > 0:
            ordered_items = [i[0] for i in sorted(local_dataset.items(),\
                        key = operator.itemgetter(1), reverse = True)]
            update_tree(ordered_items, return_tree, header_table, count)
    return return_tree, header_table

def update_tree(items, tree, header_table, count):
    if items[0] in tree.children:
        tree.children[items[0]].increase(count)
    else:
        tree.children[items[0]] = tree_node(items[0], count, tree)
    #如果这个排过序的事务的第一项（即最频繁项）在树的子集中，那么更新该项计数，如果不存在，那么创建一个新的tree_node作为子节点
        if header_table[items[0]][1] == None:
            header_table[items[0]][1] = tree.children[items[0]]
        else:
            update_link(header_table[items[0]][1], tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1:], tree.children[items[0]], header_table, count)

def update_link(test_node, target_node):
    while test_node.node_link != None:
        test_node = test_node.node_link
    test_node.node_link = target_node
    
def load_simple_data():
    simple_data = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simple_data

def data2data_count_dict(data):
    return_dict = {}
    for trans in data:
        return_dict[frozenset(trans)] = \
            return_dict.get(frozenset(trans), 0) + 1
    return return_dict

def ascend_tree(leaf_node, prefix_path):
    if leaf_node.parent != None:
        prefix_path.append(leaf_node.parent.name)
        ascend_tree(leaf_node.parent, prefix_path)
    
def find_prefix_path(base_item, tree_node):
    conditional_pattern_base = {}
    while tree_node != None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            conditional_pattern_base[frozenset(prefix_path[:-1])] = tree_node.count
        tree_node = tree_node.node_link
    return conditional_pattern_base
        
def mine_tree(FP_tree, header_table, min_support, frequent_item_list, support_data, prefix = set([])):
    #从头指针表中频度最小的项开始排序成列表
    item_list = [v[0] for v in sorted(header_table.items(), key = operator.itemgetter(1))]
    
    for base_item in item_list:
        new_frequent_set = prefix.copy()
        new_frequent_set.add(base_item)
        base_item_support_data = {}
        base_item_support_data[frozenset(new_frequent_set)] = header_table[base_item][0]
        support_data.update(base_item_support_data)
        frequent_item_list.append(frozenset(new_frequent_set))
        
        conditional_pattern_base = find_prefix_path(base_item, header_table[base_item][1])
        conditional_tree, conditional_header_table = creat_FP_tree(conditional_pattern_base, min_support)
    
        if conditional_header_table != None:
            print 'conditional tree for:', new_frequent_set
            conditional_tree.display()
            mine_tree(conditional_tree, conditional_header_table, min_support, frequent_item_list, support_data, new_frequent_set)
        
def trans(frequent_item_list, support_data, n):
    size_list = [len(i) for i in frequent_item_list]
    max_size = max(size_list)
    size = 0; trans_L = []
    while size < max_size:
        trans_L.append([])
        size += 1
    for i in range(len(frequent_item_list)):
        trans_L[size_list[i] - 1].append(frequent_item_list[i])
    
    for k, v in support_data.items():
        support_data[k] = v / float(n)
    return trans_L, support_data

def create_C_k_p_1(L_k, k):
    return_list = []
    len_L_k = len(L_k)
    for i in range(len_L_k):
        for j in range(i + 1, len_L_k):
            L1 = list(L_k[i])[: k - 1]; L2 = list(L_k[j])[: k -1]
            L1.sort(), L2.sort()
            if L1 == L2:
                return_list.append(L_k[i] | L_k[j])
    return return_list

def generate_rules(L, support_data, n, min_confidence = 0.7):
    L, support_data = trans(L, support_data.copy(), n)
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
        H_m_p_1 = create_C_k_p_1(H, m)
        H_m_p_1 = calculate_confidence(frequent_set, H_m_p_1, support_data, rule_list, min_confidence)
        if len(H_m_p_1) > 1:
            rules_from_back_element(frequent_set, H_m_p_1, support_data, rule_list, min_confidence)






