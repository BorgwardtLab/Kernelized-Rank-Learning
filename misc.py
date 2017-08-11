#!/usr/bin/env python

import os
import numpy as np
import scipy.io as sio 
import IPython as ip
import gzip

FULL, SAMPLE, KEEPK = 'FULL', 'SAMPLE', 'KEEPK'
GEX, WES, CNV, METH = 'GEX', 'WES', 'CNV', 'MET'
KRL, LKRL, KBMTL, KRR, RF, EN = 'KRL', 'LKRL', 'KBMTL', 'KRR', 'RF', 'EN'
BASELINES = [EN, RF, KRR]
METHODS = BASELINES + [KBMTL, LKRL, KRL]
PARAM_STR, RANK_STR, PERCENTILE_STR, NDCG_STR, PRECISION_STR = 'PARAM', 'RANK', 'PERCENTILE', 'NDCG', 'PRECISION'
DELIM = ' '


def read_FULL(data_name, seed, i, j=None):
    if j == None:
        X_file = '%s/data/%s/X/X_seed%s_cv%s.mat' % (os.getcwd(), data_name, seed, i)
        Y_file = '%s/data/%s/FULL_Y/FULL_Y_seed%s_cv%s.mat' % (os.getcwd(), data_name, seed, i)
    else:
        X_file = '%s/data/%s/X/X_seed%s_cv%s.%s.mat' % (os.getcwd(), data_name, seed, i, j)
        Y_file = '%s/data/%s/FULL_Y/FULL_Y_seed%s_cv%s.%s.mat' % (os.getcwd(), data_name, seed, i, j)
    X = sio.loadmat(X_file)
    X_train = X['X_train']
    X_test  = X['X_test']
    Y = sio.loadmat(Y_file)
    Y_train = Y['Y_train']
    Y_test  = Y['Y_test']    

    return X_train, X_test, Y_train, Y_test


def read_SAMPLE(data_name, seed, sample_ratio, i, j=None):
    if j == None:
        X_file = '%s/data/%s/X/X_seed%s_cv%s.mat' % (os.getcwd(), data_name, seed, i)
        Y_file = '%s/data/%s/SAMPLE_Y/SAMPLE_Y_seed%s_cv%s_sr%s.mat' % (os.getcwd(), data_name, seed, i, sample_ratio)
    else:
        X_file = '%s/data/%s/X/X_seed%s_cv%s.%s.mat' % (os.getcwd(), data_name, seed, i, j)
        Y_file = '%s/data/%s/SAMPLE_Y/SAMPLE_Y_seed%s_cv%s.%s_sr%s.mat' % (os.getcwd(), data_name, seed, i, j, sample_ratio)
    X = sio.loadmat(X_file)
    X_train = X['X_train']
    X_test  = X['X_test']
    Y = sio.loadmat(Y_file)
    Y_train = Y['Y_train']
    Y_test  = Y['Y_test']       

    return X_train, X_test, Y_train, Y_test       


def read_KEEPK(data_name, seed, keepk_ratio, keepk, i, j=None):
    if j == None:
        X_file = '%s/data/%s/X/X_seed%s_cv%s.mat' % (os.getcwd(), data_name, seed, i)
        Y_file = '%s/data/%s/KEEPK_Y/KEEPK_Y_seed%s_cv%s_kr%s_keep%s.mat' % (os.getcwd(), data_name, seed, i, keepk_ratio, keepk)
    else:
        X_file = '%s/data/%s/X/X_seed%s_cv%s.%s.mat' % (os.getcwd(), data_name, seed, i, j)
        Y_file = '%s/data/%s/KEEPK_Y/KEEPK_Y_seed%s_cv%s.%s_kr%s_keep%s.mat' % (os.getcwd(), data_name, seed, i, j, keepk_ratio, keepk)
    X = sio.loadmat(X_file)
    X_train = X['X_train']
    X_test  = X['X_test']
    Y = sio.loadmat(Y_file)
    Y_train = Y['Y_train']
    Y_test  = Y['Y_test']        

    return X_train, X_test, Y_train, Y_test


def non_nan_statistic(array_like, statistic):
    array_like = np.array(array_like)
    return statistic(array_like[~np.isnan(array_like)])


def non_nan_mean(array_like):
    return non_nan_statistic(array_like, np.mean)


def non_nan_std(array_like):
    return non_nan_statistic(array_like, np.std)


def dcg(y, pi, k):
    return ((2**y[pi[:k]] - 1) / np.log(range(2,2+k))).sum() if k > 0 else np.nan


def ndcg(y, pi, k):
    return dcg(y, pi, k) / dcg(y, np.argsort(y)[::-1], k) if k > 0 else np.nan


def precision(y, f, k):
    return (1.0 * np.intersect1d(np.argsort(y)[::-1][:k], np.argsort(f)[::-1][:k]).shape[0] / k) if k > 0 else np.nan


def NDCG(Y, F, k):
    n = Y.shape[0]
    ndcgk = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        ndcgk.append(ndcg(y, np.argsort(f)[::-1], min(k, y.shape[0])))
    return np.array(ndcgk)


def Precision(Y, F, k):
    n = Y.shape[0]
    precisionk = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        precisionk.append(precision(y, f, min(k, y.shape[0])))
    return np.array(precisionk)


def rank(pool, best, which=0):
    assert which >= 0
    return list(np.argsort(pool)[::-1]).index(np.argsort(best)[::-1][which]) if pool.shape[0] > 0 else np.nan


def Rank(Y, F, rank_type):
    assert rank_type in ['best_drug', 'best_prediction']
    n = Y.shape[0]
    ranks = []
    for i in range(n):
        y = Y[i]
        f = F[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        ranks.append(rank(f, y) if rank_type == 'best_drug' else rank(y, f))
    return np.array(ranks)


def percentile(y, f, which):
    assert which >= 0
    return (rank(y, f, which) / float(y.shape[0])) if (y.shape[0] > 0 and y.shape[0] > which) else np.nan


def Percentile(Y, F, k):
    n = Y.shape[0]
    percentiles = []
    for i in range(n):
        y = Y[i]
        f = F[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        percentiles.append([percentile(y, f, which) for which in range(k)])
    return np.array(percentiles)


def average_rank(Y, F, k):
    n = Y.shape[0]
    ar = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        pi = np.argsort(f)[::-1]
        y_pi = y[pi]
        tmp = np.where(y_pi==k)[0]
        ar.extend(tmp)
    return np.array(ar)

def raise_exception(*messages):
    print >> sys.stderr, 'ERROR:', ' '.join(map(str, messages))
    raise Exception

def open_file(filename, mode='r', compresslevel=9):
    if mode not in ['r', 'rb', 'a', 'ab', 'w', 'wb']: raise_exception('file mode not supported:', mode)
    if filename.endswith('.gz') or (not os.path.exists(filename) and os.path.exists(filename + '.gz')):
        #gzip automatically adds 'b' to the 'r', 'a', and 'w' modes
        return gzip.open(filename if filename.endswith('.gz') else filename + '.gz', mode, compresslevel)
    else:
        return open(filename, mode)
    
def line_split_gen(filename, delim='\t', strip='\n', comment='#', skip_rows=0, skip_columns=0):
    with open_file(filename) as f:
        for _ in range(skip_rows):
            f.readline()
        for line in f:
            if comment is not None and line.startswith(comment): continue
            line_split = line.strip(strip).split(delim)
            yield line_split[skip_columns:]

def cm_to_inch(value):
    return float(value) / 2.54
    
