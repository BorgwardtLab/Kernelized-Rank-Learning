import numpy as np
from sklearn.cross_validation import train_test_split

from evaluate import NDCG, Precision
from KRL import KRL_fit, KRL_pred

# Load example data
data = np.load('example.npz')
# X is the Gene expression data, nPatient X nGene
X = data['X']
# Y is the drug response data, nPatient X nDrug, Y suppose to be sparse
Y = data['Y']

# Optimize NDCG@k
k = 10
# Regularization parameter
Lambda = 0.0001
# RBF kernel parameter
Gamma = 0.0001
# Number of jobs for parallel computing
njobs = 1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=10000)
W = KRL_fit(X_train, Y_train, k, Lambda, Gamma, njobs)
Y_pred = KRL_pred(W, X_train, X_test, Gamma)

ndcgk = NDCG(Y_test, Y_pred, k)
preck = Precision(Y_test, Y_pred, k)
print 'NDCG%s: %s, PREC%s: %s' % (k, np.mean(ndcgk), k, np.mean(preck))