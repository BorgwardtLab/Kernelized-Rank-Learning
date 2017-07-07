import numpy as np
from cvxopt import matrix, solvers
from scipy import optimize

solvers.options['show_progress'] = False

def solve_Wt_kernel(A, b, t, Lambda, K_inv):
    A1 = A[1:t+1]
    n, m = A1.shape[1], A1.shape[2]
    P = np.zeros([t, t])
    for i in range(m):
        P += np.dot(np.dot(A1[:, :, i], K_inv), A1[:, :, i].T)
    P = matrix(P / Lambda)
    q = - matrix(b[1:t+1])
    G = matrix(-1*np.eye(t))
    h = matrix(np.zeros(t))
    AA = matrix(1.0, (1, t))
    bb = matrix(1.0)
    sol = solvers.qp(P, q, G, h, AA, bb)     
    alphat = np.ravel(sol['x'])
    u = - np.dot(A1.T, alphat).T / Lambda
    Wt = np.zeros([n, m])
    for i in range(m):
        Wt[:, i] = np.dot(u[:, i], K_inv)
    return Wt    

def bmrm_kernel(W0, loss, loss_gradient, args, Lambda, K, K_inv, FTOL=1e-4, MAX_ITER=1000):
    [n, m] = W0.shape
    W = np.zeros([MAX_ITER, n, m])
    W[0] = W0
    A = np.zeros([MAX_ITER+1, n, m])
    b = np.zeros([MAX_ITER+1])
    fval = np.zeros(MAX_ITER+1)
    b[1], A[1] = loss_gradient(W[0], *args)
    fval[0] = b[1] + 0.5*Lambda*np.dot(np.dot(W[0].T, K), W[0]).trace()
    b[1] = b[1] - np.multiply(A[1], W[0]).sum()
    for t in range(1, MAX_ITER):
        W[t] = solve_Wt_kernel(A, b, t, Lambda, K_inv)
        b[t+1], A[t+1] = loss_gradient(W[t], *args)
        reg = 0.5*Lambda*np.dot(np.dot(W[t].T, K), W[t]).trace()
        fval[t] = b[t+1] + reg
        b[t+1] = b[t+1] - np.multiply(A[t+1], W[t]).sum()
        fval_lb_t = np.multiply(A[t], W[t]).sum() + b[t] + reg
        epsilon = (fval[0:t+1] - fval_lb_t).min()
        print "Epsilon iter %s: %s" % (t, epsilon)
        if epsilon < FTOL:
            break       
    return W[t]        



