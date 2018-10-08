#!/usr/bin/env python

import os
import sys
import yaml
import numpy as np
import scipy.io as sio

from sklearn.model_selection import KFold

from misc import FULL, SAMPLE, KEEPK
from misc import GEX, WES, CNV, METH

def sample(Y_train, sample_ratio):
    Y_train_sample = np.copy(Y_train)
    for i in range(Y_train_sample.shape[0]):
        index = np.random.binomial(1, sample_ratio, size=Y_train_sample.shape[1])
        Y_train_sample[i, np.where(index==0)] = np.nan 
    return Y_train_sample

def keepk_sample(Y_train, Y_test, keepk_ratio, keepk):
    Y_train_keepk = np.copy(Y_train)
    N = int(keepk_ratio * Y_train_keepk.shape[1]) 
    for i in range(Y_train_keepk.shape[0]):
        y = Y_train_keepk[i]
        notnan = ~np.isnan(y)
        y = y[notnan]
        y_argsort = np.argsort(y)[::-1]
        y_argsort_pos = y_argsort[:N]
        y_argsort_neg = y_argsort[N:]
        pos_permutation = np.random.permutation(y_argsort_pos.shape[0])
        for j in range(keepk, pos_permutation.shape[0]):
            y[y_argsort_pos[pos_permutation[j]]] = np.nan
        neg_permutation = np.random.permutation(y_argsort_neg.shape[0])
        for j in range(0, neg_permutation.shape[0]):
            y[y_argsort_neg[neg_permutation[j]]] = np.nan
        Y_train_keepk[i, notnan] = y       

    keep = [] 
    for i in range(Y_train_keepk.shape[1]):
        y = Y_train_keepk[:, i]
        if y[~np.isnan(y)].shape[0] > 5:
            keep.append(i)
    keep = np.array(keep)

    Y_train_keepk = Y_train_keepk[:, keep]
    Y_test_keepk = Y_test[:, keep]   

    return Y_train_keepk, Y_test_keepk   

def main():
    print 'Splitting the data for training and testing, creating folds for cross-validation...'

    config_file = sys.argv[1] 
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    seeds = np.array(config['seeds'], dtype=int)
    analysis = config['analysis']
    assert analysis in [FULL, SAMPLE, KEEPK], 'Unknown analysis type %s specified in the config file' % analysis
    data_name = config['data']
    assert data_name in [GEX, WES, CNV, METH], 'Unknown data type %s specified in the config file' % data_name
    cv = config['cv']
    sample_ratios = np.array(config['sample_ratios'], dtype=float)
    keepk_ratios = np.array(config['keepk_ratios'], dtype=float)
    keepk = config['keepk']

    directory = '%s/data/%s' % (os.getcwd(), data_name)
    if not os.path.exists(directory):
        os.makedirs(directory)    

    data = np.load('%s/data/GDSC_%s.npz' % (os.getcwd(), data_name))
    X = data['X']
    Y = data['Y']         

    directory = '%s/data/%s/X' % (os.getcwd(), data_name)
    if not os.path.exists(directory):
        os.makedirs(directory)  

    for seed in seeds:
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train = X[train_index]                
            X_test = X[test_index]
            out = {}
            out['X_train'] = X_train
            out['X_test'] = X_test              
            sio.savemat('%s/X_seed%s_cv%s.mat' % (directory, seed, i), out)
            for j, (train_index, test_index) in enumerate(kf.split(X_train)):
                X_train = X[train_index]          
                X_test = X[test_index]
                out = {}
                out['X_train'] = X_train
                out['X_test'] = X_test                  
                sio.savemat('%s/X_seed%s_cv%s.%s.mat' % (directory, seed, i, j), out)     

    directory = '%s/data/%s/%s_Y' % (os.getcwd(), data_name, analysis)
    if not os.path.exists(directory):
        os.makedirs(directory)  

    if analysis == FULL:
        for seed in seeds:
            kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
            for i, (train_index, test_index) in enumerate(kf.split(Y)):
                Y_train = Y[train_index]                
                Y_test = Y[test_index]
                out = {}
                out['Y_train'] = Y_train
                out['Y_test'] = Y_test                
                sio.savemat('%s/FULL_Y_seed%s_cv%s.mat' % (directory, seed, i), out)
                for j, (train_index, test_index) in enumerate(kf.split(Y_train)):
                    Y_train = Y[train_index]                
                    Y_test = Y[test_index]
                    out = {}
                    out['Y_train'] = Y_train
                    out['Y_test'] = Y_test                     
                    sio.savemat('%s/FULL_Y_seed%s_cv%s.%s.mat' % (directory, seed, i, j), out)     

    if analysis == SAMPLE:
        for seed in seeds:
            np.random.seed(seed)
            kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
            for sr in sample_ratios:
                for i, (train_index, test_index) in enumerate(kf.split(Y)):
                    Y_train = Y[train_index]                
                    Y_test = Y[test_index]
                    Y_train_sample = sample(Y_train, sr)
                    out = {}
                    out['Y_train'] = Y_train_sample
                    out['Y_test'] = Y_test  
                    sio.savemat('%s/SAMPLE_Y_seed%s_cv%s_sr%s.mat' % (directory, seed, i, sr), out)
                    for j, (train_index, test_index) in enumerate(kf.split(Y_train)):
                        Y_train = Y[train_index]                
                        Y_test = Y[test_index]
                        Y_train_sample = sample(Y_train, sr)
                        out = {}
                        out['Y_train'] = Y_train_sample
                        out['Y_test'] = Y_test  
                        sio.savemat('%s/SAMPLE_Y_seed%s_cv%s.%s_sr%s.mat' % (directory, seed, i, j, sr), out)

    if analysis == KEEPK:
        for seed in seeds:
            np.random.seed(seed)
            kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
            for kr in keepk_ratios:
                for i, (train_index, test_index) in enumerate(kf.split(Y)):
                    Y_train = Y[train_index]
                    Y_test = Y[test_index]
                    Y_train_keepk, Y_test_keepk = keepk_sample(Y_train, Y_test, kr, keepk)
                    out = {}
                    out['Y_train'] = Y_train_keepk
                    out['Y_test'] = Y_test_keepk  
                    sio.savemat('%s/KEEPK_Y_seed%s_cv%s_kr%s_keep%s.mat' % (directory, seed, i, kr, keepk), out)  
                    for j, (train_index, test_index) in enumerate(kf.split(Y_train)):
                        Y_train = Y[train_index]
                        Y_test = Y[test_index]
                        Y_train_keepk, Y_test_keepk = keepk_sample(Y_train, Y_test, kr, keepk)
                        out = {}
                        out['Y_train'] = Y_train_keepk
                        out['Y_test'] = Y_test_keepk
                        sio.savemat('%s/KEEPK_Y_seed%s_cv%s.%s_kr%s_keep%s.mat' % (directory, seed, i, j, kr, keepk), out)
    print 'Finished.'

if __name__ == '__main__':
    main()
