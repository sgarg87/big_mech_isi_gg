import numpy as np
import time
import sys
from config import *


def normalize_prob_sum_1_col(T):
    n = T.shape[0]
    assert T.shape[0] == T.shape[1] == n
    T /= np.tile(T.sum(0), (n, 1)).astype(np.float)
    return T


def normalize_prob_sum_1_row(Y):
    n = Y.shape[0]
    k = Y.shape[1]
    Y /= np.tile(Y.sum(1).reshape(n, 1), (1, k)).astype(np.float)
    return Y


def get_transition_matrix_frm_kernel(K):
    # assuming that kernel values is normalized, thereby giving maximum kernel value "1" along diagonal.
    assert np.all(0 <= K) and np.all( K <= 1)
    T = normalize_prob_sum_1_col(K)
    K = None
    return T


def get_transition_matrix_frm_distance(D):
    assert np.all(D >= 0)
    T = np.exp(-D)
    D = None
    T = normalize_prob_sum_1_col(T)
    return T


def infer_labels_fr_test(K, D, labels_train, train, test, tol=1e-3, num_lp_iter=100):
    #
    # Learning from labeled and unlabeled data with label propagation; Zhu and Ghahramani; NIPS
    #
    # K is kernel matrix
    # D is distance matrix
    # either K or D should not be None
    # labels train is labels for training set (numpy array)
    # train is array of indices for training in the weight matrix
    # test is array of indices for testing in the weight matrix
    #
    n = train.size
    m = test.size
    assert n == labels_train.size
    assert K is not None or D is not None
    assert not(K is not None and D is not None)
    if K is not None:
        T = get_transition_matrix_frm_kernel(K)
    elif D is not None:
        T = get_transition_matrix_frm_distance(D)
    else:
        raise AssertionError
    del D, K
    #
    assert T.shape[0] == T.shape[1] == (m+n)
    #
    # initialize probability for labels
    #assuming standard three labels 0-negative,1-positive,2-swap
    #
    alpha = np.ones(3)/3
    Y = np.random.dirichlet(alpha, size=(m+n))
    start_time = time.time()
    print '\n'
    for curr_idx in range(num_lp_iter):
        sys.stdout.write('*')
        sys.stdout.flush()
        Y_old = np.copy(Y)
        Y = np.dot(T, Y)
        Y = normalize_prob_sum_1_row(Y)
        Y[train, :] = 0
        Y[train, labels_train] = 1
        #
        if np.all(abs(Y-Y_old) < tol):
            break
    compute_time = time.time()-start_time
    if debug:
        print 'Time for this label propagation iteration was {}'.format(compute_time)
    return Y[test]
