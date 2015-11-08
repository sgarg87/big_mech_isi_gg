import constants_absolute_path as cap
import pickle
import save_sparse_scipy_matrices as sssm
import numpy as np
import time
import random as r
r.seed(871227)
import scipy.sparse.linalg as ssl


def get_chicago_test_data_idx(amr_graphs_org):
    n = amr_graphs_org.shape[0]
    assert amr_graphs_org.shape[1] == 1
    chicago_idx_list = []
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs_org[curr_idx, 0]
        #
        curr_path = curr_amr_graph_map['path']
        #
        if 'chicago' in curr_path:
            chicago_idx_list.append(curr_idx)
    #
    chicago_idx_list = np.array(chicago_idx_list)
    return chicago_idx_list


def get_amr_data():
    amr_pickle_file_path = './amr_data_temp.pickle'
    with open(cap.absolute_path+amr_pickle_file_path, 'rb') as f_p:
        amr_data = pickle.load(f_p)
        amr_graphs = amr_data['amr']
        labels = amr_data['label']
        return amr_graphs, labels


def train_gpc(K_train, labels_train):
    file_path = './lst_sqr_kernel_Kinv_mul_Y'
    print 'Learning Gaussian process classifications ...'
    #
    c = 3
    score_train = -c*np.ones(labels_train.shape)
    score_train[np.where(labels_train == 1)] = c
    print 'score_train', score_train
    labels_train = None
    #
    start_time = time.time()
    print 'computing the least squares'
    assert K_train.shape[0] == K_train.shape[1]
    #
    # todo: this lsqr algorithm is parallelizable, so do the needful
    # see the classical paper LSQR An algrithm for sparse linear equations and sparse least squares.pdf
    #
    x = ssl.lsqr(K_train, score_train, show=True)
    np.savez_compressed(file_path, x)
    print 'Time to compute the least square solution was ', time.time()-start_time
    K_train = None
    #
    return x


if __name__ == '__main__':
    is_load_classifier = False
    #
    amr_graphs, labels = get_amr_data()
    #
    n = labels.size
    print 'labels.size', labels.size
    #
    idx_label2 = np.where(labels == 2)
    labels[idx_label2] = 0
    #
    test = get_chicago_test_data_idx(amr_graphs)
    print 'test.shape', test.shape
    #
    train = np.setdiff1d(np.arange(0, n), test)
    print 'train.shape', train.shape
    #
    test = None
    amr_graphs = None
    #
    if not is_load_classifier:
        labels_train = labels[train]
        print 'labels_train.shape', labels_train.shape
    else:
        labels_train = None
    #
    k_path = './graph_kernel_matrix_joint_train_data_parallel/num_cores_100.npz'
    K = sssm.load_sparse_csr(cap.absolute_path+k_path)
    print 'K.shape', K.shape
    #
    if not is_load_classifier:
        K_train = K[train, :]
        K_train = K_train.tocsc()
        K_train = K_train[:, train]
        K_train = K_train.tocsr()
        print 'K_train.shape', K_train.shape
        print 'K_train.nnz', K_train.nnz
    else:
        K_train = None
    #
    train = None
    K = None
    #
    print 'Gaussian process ...'
    error = train_gpc(K_train, labels_train)
