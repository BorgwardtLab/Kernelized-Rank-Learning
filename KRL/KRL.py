import numpy as np

from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed

from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from bundle_method_kernel import bmrm_kernel


def ndcgk_vector_loss_gradient(y, f, k):
    if y.shape[0] < k:
        k = y.shape[0]
    m = len(y)
    a = np.zeros(m)
    a[:k] = 1.0 / np.log(np.arange(2, k+2))
    b = 2**y - 1
    c = np.arange(1, m+1)**(-0.25)
    C = np.outer(a, b) - np.outer(c, f)
    pi = linear_sum_assignment(C)[1]
    loss = np.dot(f[pi] - f, c) + np.dot(a - a[pi], b)
    pi_inverse = np.argsort(pi)
    gradient = c[pi_inverse] - c
    return loss, gradient

def ndcgk_block_loss_gradient(index, F, Y, k, Notnan):
    [n, m] = Y.shape
    l = 0
    g = np.zeros(F.shape)    
    for i in range(n):
        f, y = F[i, Notnan[i]], Y[i, Notnan[i]]
        order = np.argsort(y)[::-1]
        back_order = np.argsort(order)
        loss, gradient = ndcgk_vector_loss_gradient(y[order], f[order], k)
        l += loss
        g[i, Notnan[i]] = gradient[back_order]
    return index, l, g

def ndcgk_row_loss_gradient(W, X, Y, k, Notnan, njobs):
    [n, m] = Y.shape
    [n, p] = X.shape
    W = W.reshape([p, m])
    l = 0
    F = np.dot(X, W)
    g = np.zeros(F.shape)


    step = n / njobs
    if step == 0:
        step = 1
    results = Parallel(n_jobs=njobs)(delayed(ndcgk_block_loss_gradient)(i, F[i:i+step], Y[i:i+step], k, Notnan[i:i+step]) for i in xrange(0, n, step))    
    for i ,li, gi in results:
        l += li
        g[i:i+step] = gi

    g = np.dot(X.T, g)

    return l, g

def ndcgk_block_loss(F, Y, k, Notnan):
    [n, m] = Y.shape
    l = 0
    for i in range(n):
        f, y = F[i, Notnan[i]], Y[i, Notnan[i]]
        order = np.argsort(y)[::-1]
        back_order = np.argsort(order)
        loss, gradient = ndcgk_vector_loss_gradient(y[order], f[order], k)
        l += loss
    return l

def ndcgk_row_loss(W, X, Y, k, Notnan, njobs):
    [n, m] = Y.shape
    [n, p] = X.shape
    W = W.reshape([p, m])
    l = 0
    F = np.dot(X, W)

    step = n / njobs
    if step == 0:
        step = 1    
    results = Parallel(n_jobs=njobs)(delayed(ndcgk_block_loss)(F[i:i+step], Y[i:i+step], k, Notnan[i:i+step]) for i in xrange(0, n, step))    
    for li in results:
        l += li

    return l

def KRL_fit(X, Y, k, Lambda, gamma, njobs, verbose=True):
    if gamma == 'linear':
        K = linear_kernel(X)
    else:
        K = rbf_kernel(X, gamma=gamma)
    K_inv = np.linalg.inv(K + 1e-6*np.eye(K.shape[0], K.shape[0]))    

    [n, p] = K.shape
    [n, m] = Y.shape
    Notnan_row = []
    for i in range(n):
        y = Y[i]
        Notnan_row.append(~np.isnan(y))

    np.random.seed(0)
    W = 0.01*np.random.randn(p, m)
    W = bmrm_kernel(W, ndcgk_row_loss, ndcgk_row_loss_gradient, (K, Y, k, Notnan_row, njobs), Lambda, K, K_inv, MAX_ITER=1000, verbose=verbose)
    W = W.reshape([p, m])   

    return W

def KRL_pred(W, X_train, X_test, gamma):
    if gamma == 'linear':
        K = linear_kernel(X_test, X_train)
    else:
        K = rbf_kernel(X_test, X_train, gamma=gamma)
    return np.dot(K, W)

def KRL(X_train, Y_train, X_test, k, Lambda, gamma, njobs, verbose=True):
    W = KRL_fit(X_train, Y_train, k, Lambda, gamma, njobs, verbose)
    Y_pred = KRL_pred(W, X_train, X_test, gamma)
    return Y_pred

