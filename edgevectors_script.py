import constants_absolute_path as cap
import pickle
import edge_labels_propagation as elp
import numpy as np
import matplotlib.pyplot as plt
import math

#
is_amr_edge2vec = True
if is_amr_edge2vec:
    amr_edge2vec_map = None
#
is_dependencies2vec = True
if is_dependencies2vec:
    dependencies_edge2vec_map = None
#
if is_amr_edge2vec:
    print 'Loading amr edge vectors ...'
    amr_edge_label_word_vec_pickled_file_path = elp.get_edge_label_wordvector_file_path(is_amr=True)
    with open(cap.absolute_path+amr_edge_label_word_vec_pickled_file_path, 'r') as f_amr_edge_word2vec:
        amr_edge2vec_map = pickle.load(f_amr_edge_word2vec)
if is_dependencies2vec:
    print 'loading dependencies edge vectors ...'
    dependencies_edge_label_word_vec_pickled_file_path = elp.get_edge_label_wordvector_file_path(is_amr=False)
    with open(cap.absolute_path+dependencies_edge_label_word_vec_pickled_file_path, 'r') as f_dependencues_edge_word2vec:
        dependencies_edge2vec_map = pickle.load(f_dependencues_edge_word2vec)


def filter_edge_labels_inverse(edge_labels_list):
    new_edge_labels_list = []
    for curr_edge_label in edge_labels_list:
        if not curr_edge_label.endswith('-of'):
            new_edge_labels_list.append(curr_edge_label)
    print 'new_edge_labels_list', new_edge_labels_list
    return new_edge_labels_list


def compute_amr_edge_similarity_matrix():
    amr_edge_labels_list = amr_edge2vec_map.keys()
    amr_edge_labels_list = filter_edge_labels_inverse(amr_edge_labels_list)
    amr_edge_labels_list.sort()
    n = len(amr_edge_labels_list)
    m = 10000
    X = np.zeros((n,m))
    for i in range(n):
        curr_edge_label = amr_edge_labels_list[i]
        curr_edge_vector = amr_edge2vec_map[curr_edge_label]
        assert curr_edge_vector.size == m
        X[i] = curr_edge_vector
    #
    A = np.dot(X, X.transpose())
    #
    print A.shape
    norm = np.sqrt(A.diagonal().repeat(n).reshape((n, n)))
    A /= norm*norm.transpose()
    #
    plt.pcolor(A, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(1, n+1), amr_edge_labels_list, rotation='vertical', size='5')
    plt.yticks(range(1, n+1), amr_edge_labels_list, size='5')
    plt.savefig('./amr_edge_vectors_similarity.pdf', dpi=3000, format='pdf')
    plt.close()


def compute_sdg_edge_similarity_matrix():
    #
    sdg_edge_labels_list = dependencies_edge2vec_map.keys()
    sdg_edge_labels_list = filter_edge_labels_inverse(sdg_edge_labels_list)
    sdg_edge_labels_list.sort()
    n = len(sdg_edge_labels_list)
    m = 10000
    X = np.zeros((n,m))
    for i in range(n):
        curr_edge_label = sdg_edge_labels_list[i]
        curr_edge_vector = dependencies_edge2vec_map[curr_edge_label]
        assert curr_edge_vector.size == m
        X[i] = curr_edge_vector
    #
    A = np.dot(X, X.transpose())
    #
    print A.shape
    norm = np.sqrt(A.diagonal().repeat(n).reshape((n, n)))
    A /= norm*norm.transpose()
    #
    plt.pcolor(A, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(1, n+1), sdg_edge_labels_list, rotation='vertical', size='2')
    plt.yticks(range(1, n+1), sdg_edge_labels_list, size='2')
    plt.savefig('./sdg_edge_vectors_similarity.pdf', dpi=3000, format='pdf')
    plt.close()


def compute_amr_sdg_edge_similarity_matrix():
    #
    sdg_edge_labels_list = dependencies_edge2vec_map.keys()
    sdg_edge_labels_list = filter_edge_labels_inverse(sdg_edge_labels_list)
    sdg_edge_labels_list.sort()
    ns = len(sdg_edge_labels_list)
    #
    amr_edge_labels_list = amr_edge2vec_map.keys()
    amr_edge_labels_list = filter_edge_labels_inverse(amr_edge_labels_list)
    amr_edge_labels_list.sort()
    na = len(amr_edge_labels_list)
    #
    m = 10000
    #
    Xs = np.zeros((ns, m))
    for i in range(ns):
        curr_edge_label = sdg_edge_labels_list[i]
        curr_edge_vector = dependencies_edge2vec_map[curr_edge_label]
        assert curr_edge_vector.size == m
        Xs[i] = curr_edge_vector
    #
    Xa = np.zeros((na, m))
    for i in range(na):
        curr_edge_label = amr_edge_labels_list[i]
        curr_edge_vector = amr_edge2vec_map[curr_edge_label]
        assert curr_edge_vector.size == m
        Xa[i] = curr_edge_vector
    #
    A = np.zeros((na, ns))
    for i in range(na):
        for j in range(ns):
            curr_amr_edge_vector = Xa[i]
            curr_sdg_edge_vector = Xs[j]
            ij = np.dot(curr_amr_edge_vector, curr_sdg_edge_vector.transpose())
            ii = np.dot(curr_amr_edge_vector, curr_amr_edge_vector.transpose())
            jj = np.dot(curr_sdg_edge_vector, curr_sdg_edge_vector.transpose())
            #
            ij_norm = ij/math.sqrt(ii*jj)
            print 'ij_norm', ij_norm
            A[i,j] = ij_norm
    #
    A = A.transpose()
    #
    plt.pcolor(A, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(1,na+1), amr_edge_labels_list, rotation='vertical', size='5')
    plt.yticks(range(1,ns+1), sdg_edge_labels_list, size='2')
    plt.savefig('./amr_sdg_edge_vectors_similarity.pdf', dpi=3000, format='pdf')
    plt.close()


if __name__ == '__main__':
    compute_amr_edge_similarity_matrix()
    compute_sdg_edge_similarity_matrix()
    compute_amr_sdg_edge_similarity_matrix()
