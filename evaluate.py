import numpy as np

def dcg(y, pi, k):
    return ((2**y[pi[:k]] - 1) / np.log(range(2,2+k))).sum()

def ndcg(y, pi, k):
    return dcg(y, pi, k) / dcg(y, np.argsort(y)[::-1], k)    

def precision(y, f, k):
    return 1.0 * np.intersect1d(np.argsort(y)[::-1][:k], np.argsort(f)[::-1][:k]).shape[0] / k

def NDCG(Y, F, k):
    n = Y.shape[0]
    ndcgk = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        if k > 0 and y.shape[0] > k and np.unique(y).shape[0] > 1:
            ndcgk.append(ndcg(y, np.argsort(f)[::-1], k))
    return np.array(ndcgk)

def Precision(Y, F, k):
    n = Y.shape[0]
    precisionk = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]  
        if k > 0 and y.shape[0] > k and np.unique(y).shape[0] > 1:
            precisionk.append(precision(y, f, k))
    return np.array(precisionk)

