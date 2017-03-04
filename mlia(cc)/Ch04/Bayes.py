# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 15:54:18 2016

@author: apple
"""
import numpy as np
import re

def load_dataSet():
    posting_list=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return posting_list, class_vec

def creat_vocabulary_list(data_set):
    vocabulary_set = set([])
    for document in data_set:
        vocabulary_set = vocabulary_set | set(document)
    return list(vocabulary_set)

#set-of-words model词集模型
def set_of_words2vec(vocabulary_list, input_set):
    return_vec = np.zeros(len(vocabulary_list))
    for word in input_set:
        if word in vocabulary_list:
            return_vec[vocabulary_list.index(word)] = 1
    return return_vec

#bag-of-words model 词袋模型
def bag_of_words2vec(vocabulary_list, input_set):
    return_vec = np.zeros(len(vocabulary_list))
    for word in input_set:
        if word in vocabulary_list:
            return_vec[vocabulary_list.index(word)] += 1
    return return_vec
    
def words_list2mat(words_list, vocabulary_list):
    n = len(words_list)
    return_mat = np.array(())
    for i in words_list:
        vec = set_of_words2vec(vocabulary_list, i)
        return_mat = np.r_[return_mat, vec]
    return_mat = return_mat.reshape((n, -1))
    return return_mat

def train_NB0(X, y):
    n, m = X.shape
    p1 = sum(y)/float(n)
    p0_num = np.ones(m); p1_num = np.ones(m)
    p0_denom = 2.; p1_denom = 2.
    for i in range(n):
        if y[i] == 1:
            p1_num += X[i]
            p1_denom += sum(X[i])
        else:
            p0_num += X[i]
            p0_denom += sum(X[i])
    p1_vec = np.log(p1_num / p1_denom); p0_vec = np.log(p0_num / p0_denom)
    return p0_vec, p1_vec, p1

def classifyNB(X, p0_vec, p1_vec, p1):
    p0 = np.sum(X * p0_vec, axis = 1) + np.log(1 - p1)
    p1 = np.sum(X * p1_vec, axis = 1) + np.log(p1)
    return (p1 > p0).astype(int)

def testingNB():
    words_list, y = load_dataSet()
    vocabulary_list = creat_vocabulary_list(words_list)
    X = words_list2mat(words_list, vocabulary_list)
    p0_vec, p1_vec, p1 = train_NB0(X, y)
    
    test_entry = ['love', 'my', 'dalmation']
    x = set_of_words2vec(vocabulary_list, test_entry)
    print test_entry, 'classified as:', classifyNB(x, p0_vec, p1_vec, p1)
    test_entry = ['stupid', 'garbage']
    x = set_of_words2vec(vocabulary_list, test_entry)
    print test_entry, 'classified as:', classifyNB(x, p0_vec, p1_vec, p1)

def text_parse(text):
    tokens = re.split('\W*', text)
    return [tok.lower() for tok in tokens if len(tok) > 2]

def spam_test(seed = 1031):
    words_list = []; y = []
    for i in range(1, 26):
        tokens = text_parse(open('D:/mlia(cc)/Ch04/email/spam/%d.txt' %i).read())
        words_list.append(tokens)
        y.append(1)
        tokens = text_parse(open('D:/mlia(cc)/Ch04/email/ham/%d.txt' %i).read())
        words_list.append(tokens)
        y.append(0)
    vocabulary_list = creat_vocabulary_list(words_list)
    words_list = np.array(words_list); y = np.array(y)
    
    np.random.seed(seed); r = range(50); np.random.shuffle(r)
    X = words_list2mat(words_list, vocabulary_list)
    X_train = X[r[: 40]]; X_test = X[r[40: ]]
    y_train = y[r[: 40]]; y_test = y[r[40: ]]
    
    p0_vec, p1_vec, p1 = train_NB0(X_train, y_train)
    
    y_predict = classifyNB(X_test, p0_vec, p1_vec, p1)
    error_rate = 1 - np.sum(y_predict == y_test)/10.
    print 'the error rate is :', error_rate
    return error_rate
    
def most_freq(vocabulary_list, full_text, n = 30):
    import operator
    freq_dictionary = {}
    for token in vocabulary_list:
        freq_dictionary[token]=full_text.count(token)
    sorted_freq = sorted(freq_dictionary.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sorted_freq[:n]

def local_words(feed1, feed0, f = 30):
    words_list = []; y = []; full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        tokens = text_parse(feed1['entries'][i]['summary'])
        words_list.append(tokens)
        full_text.extend(tokens)
        y.append(1)
        tokens = text_parse(feed0['entries'][i]['summary'])
        words_list.append(tokens)
        full_text.extend(tokens)
        y.append(0)
    vocabulary_list = creat_vocabulary_list(words_list)
    words_list = np.array(words_list); y = np.array(y)
    top30_words = most_freq(vocabulary_list, full_text, n = f)
    #words_list, y, full_text, top30_words
    for pair in top30_words:
        vocabulary_list.remove(pair[0])
    
    n = 2*min_len
    n_train = int(n * 0.8); n_test = n - n_train
#    np.random.seed(1031); 
    r = range(n); np.random.shuffle(r)
    X = words_list2mat(words_list, vocabulary_list)
    X_train = X[r[: n_train]]; X_test = X[r[n_train: ]]
    y_train = y[r[: n_train]]; y_test = y[r[n_train: ]]
    p0_vec, p1_vec, p1 = train_NB0(X_train, y_train)

    y_predict = classifyNB(X_test, p0_vec, p1_vec, p1)
    error_rate = 1 - np.sum(y_predict == y_test)/float(n_test)
    print 'the error rate is : %8f' %error_rate
#    print y_predict, y_test
    return vocabulary_list, p0_vec, p1_vec
#    return error_rate

def get_top_words(ny, sf, n = 15):
    vocabulary_list, p0_vec, p1_vec = local_words(ny, sf)
    vocabulary_list = np.array(vocabulary_list)        
    sorted_sf = vocabulary_list[p0_vec.argsort()][:n]
    sorted_ny = vocabulary_list[p1_vec.argsort()][:n]
    print "SF**SF**SF**SF**SF**"
    for item in sorted_sf:
        print item
    print "NY**NY**NY**NY**NY**"
    for item in sorted_ny:
        print item


    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    