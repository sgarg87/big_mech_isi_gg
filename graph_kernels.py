# ##############################################################################
# Kernel methods for comparison of AMRs
# Reference: Kernel Methods for Relation Extraction; JMLR; 2003
# ##############################################################################

import copy
import numpy as np
import extract_from_amr_dot as ead
# import matplotlib.pyplot as plt
import util as my_util
from constants_amrs import *
import sklearn.svm as skl_svm
import sklearn.metrics as skm
import sklearn.cluster as skc
# import corex_private.corex as crx
import numpy.random as rnd
import math
from wordvec import *
import scipy.signal as ss
import constants_absolute_path as cap
from config_console_output import *
import config_hpcc as ch


#constants
const_same_kernel_call_coeff = 'const_same_kernel_call_coeff' #for the kernel computations corresponding to cycles, the coefficient terms are stored


def get_cosine_similarity_kernel_melkumyan(cs, ct):
    raise NotImplementedError #since it gives error in kernel validity
    #this function seems invalid, it does not give positive definitiveness if ct -> 0.1 (even for ct approx. 0.3 if expo variable is high)
    #Reference: A sparse covariance function for exact Gaussian process inference in large data sets; IJCAI; 2009
    #cs is cosine similarity
    #cs and ct are scalars
    cs = float(cs)
    ct = float(ct)
    r = ct + 1 - cs
    expo = float(2.5) #this is an additional step to the standard kernel
    r = r**expo
    if cs > ct:
        k = ((2 + math.cos(2*math.pi*r))/3)*(1-r)
        k += (1/(2*math.pi))*math.sin(2*math.pi*r)
        r_ = ct**expo
        k /= ((2 + math.cos(2*math.pi*r_))/3)*(1-r_) + ((1/(2*math.pi))*math.sin(2*math.pi*r_)) #normalization to get kernel similarity 1 for self
    else:
        k = 0
    return k


def get_cosine_similarity_kernel_rbf_sparse(cs, ct):
    #Reference: Compactly Supported Radial Basis Function Kernels; Hao Helen Zhang and Marc Genton
    #todo: exetend for array input of cs
    #todo: also use v parameter, v >= (d+1)/2 where d is dimensionality of word vectors
    #in our case, disregarding dimensionality of word vectors (since we use cosine similarity directly to evaluate distance), we use v = 1
    cs = float(cs)
    ct = float(ct)
    if cs < ct:
        return 0
    k = math.exp((cs-1)/rbf_bandwidth)
    k *= get_cosine_similarity_kernel_sparse(cs, ct)
    # todo: if frequency parameter is used in rbf or the sparse function, the kernel values would have to be normalized (esp. because kernel value between two same words is assumed to be one by default)
    return k


def get_cosine_similarity_kernel_sparse(cs, ct):
    k_s = 1-((1-cs)/(1-ct)) #sparse component
    if k_s < 0:
        k_s = 0
    k_s = k_s**sparse_kernel_v
    return k_s


def get_cosine_similarity_kernel(cs, ct):
    #todo: exetend for array input of cs
    if word_kernel == 'rbf_sparse':
        k = get_cosine_similarity_kernel_rbf_sparse(cs, ct)
    else:
        raise AssertionError
    if not ((0-1e-2) <= k <= (1+1e-2)):
        print 'cosine similarity kernel value {} is not in valid range.'.format(k)
        raise AssertionError
    return k


def get_cosine_similarity(word1, word2, word1_vec, word2_vec):
    word1_lower = word1.lower()
    word2_lower = word2.lower()
    if debug:
        print 'evaluating cosine similarity between {} and {}'.format(word1, word2)
        print 'evaluating cosine similarity between {} and {}'.format(word2, word1)
    if (word1_lower, word2_lower) not in cosine_similarity_map:
        cs_12 = np.dot(word1_vec, word2_vec)
        cs_11 = np.dot(word1_vec, word1_vec)
        cs_22 = np.dot(word2_vec, word2_vec)
        assert cs_11 >= 0, str(cs_11)
        assert cs_22 >= 0, str(cs_22)
        if debug:
            print 'cs_12 ', cs_12
            print 'cs_11', cs_11
            print 'cs_22', cs_22
        # todo: partition function should be stored in a separate map
        normalized_cs_12 = cs_12/math.sqrt(cs_11*cs_22)
        if debug:
            print 'normalized_cs_12 ', normalized_cs_12
        if not ((-1-1e-2) <= normalized_cs_12 <= (1+1e-2)):
            print 'normalized_cs_12 {} is out of valid range.'.format(normalized_cs_12)
            raise AssertionError
        cosine_similarity_map[(word1_lower, word2_lower)] = normalized_cs_12
    return cosine_similarity_map[(word1_lower, word2_lower)]


def test_word_kernel_pos_def(ct):
    raise DeprecationWarning
    num_trials = 10
    num_words = 100
    vocab = word2vec_model.vocab
    print 'vocab size is ', vocab.shape[0]
    for i in range(num_trials):
        words_list = word2vec_model.vocab[rnd.randint(0, vocab.shape[0]-1, num_words)]
        K = -1*np.ones((num_words, num_words))
        for l in range(num_words):
            for m in range(num_words):
                K[l, m] = get_cosine_similarity_kernel(get_cosine_similarity(words_list[l], words_list[m], get_wordvector(words_list[l]), get_wordvector(words_list[m])), ct)
        print 'K', K
        # K += reg_lambda*np.eye(num_words)
        # matrix_name = './gen_kernel_matrices/word_kernel_K_'
        # matrix_name += '_ct'+str(ct)+'_'+str(i)
        #plotting
        # plt.pcolor(K)
        # plt.colorbar()
        # plt.savefig(cap.absolute_path+matrix_name+'.pdf', dpi=1000)
        # plt.close()
        if not is_semi_pos_def(K):
            print 'positive definite test failed for word vectors'
            return


# def get_weighted_cosine_similarity_words_and_labels(cs_words, cs_labels):
#     #cs_words is cosine similarity defined for words
#     #cs_labels is cosine similarity defined for labels (usually 0/1)
#     return ((1-label_cosine_similarity_weight)*cs_words) + (label_cosine_similarity_weight*cs_labels)


def aperiodic_convolution(vec1, vec2):
    # Plate, Tony A. "Holographic reduced representations." Neural networks, IEEE transactions on 6.3 (1995): 623-641.
    # Plate, Tony. "Holographic Reduced Representations: Convolution Algebra for Compositional Distributed Representations." IJCAI. 1991.
    if vec1 is None or vec2 is None:
        return None
    n = vec1.shape[0]
    if n != vec2.shape[0]:
        raise AssertionError
    m = n-1
    if m % 2 == 1:
        p = -(m-1)/2
        q = (m+1)/2
    else:
        p = -m/2
        q = m/2
    t = []
    for j in range(-m, m+1):
        expr = 0
        for k in range(p, q+1):
            expr += vec1[k]*vec2[j-k]
        t.append(expr)
    return np.array(t)


def outer_product_vec(vec1, vec2):
    if vec1 is None or vec2 is None:
        raise AssertionError
        # return None
    result_vec = np.outer(vec1, vec2)
    return result_vec.reshape(result_vec.size)


def circular_convolution(vec1, vec2):
    # Plate, Tony A. "Holographic reduced representations." Neural networks, IEEE transactions on 6.3 (1995): 623-641.
    # Plate, Tony. "Holographic Reduced Representations: Convolution Algebra for Compositional Distributed Representations." IJCAI. 1991.
    raise Warning('Implementation too slow, use numpy scipy package Fast Fourier Tranform instead ...')
    if vec1 is None or vec2 is None:
        raise AssertionError
        # return None
    n = vec1.shape[0]
    if n != vec2.shape[0]:
        raise AssertionError
    t = []
    for j in range(n):
        expr = 0
        for k in range(n):
            expr += vec1[k]*vec2[j-k]
        t.append(expr)
    return np.array(t)


def circular_compression(vec1, vec2):
    raise Warning('Implementation too slow, use numpy scipy package Fast Fourier Tranform instead ...')
    # Plate, Tony A. "Holographic reduced representations." Neural networks, IEEE transactions on 6.3 (1995): 623-641.
    # Plate, Tony. "Holographic Reduced Representations: Convolution Algebra for Compositional Distributed Representations." IJCAI. 1991.
    if vec1 is None or vec2 is None:
        raise AssertionError
        # return None
    n = vec1.shape[0]
    if n != vec2.shape[0]:
        raise AssertionError
    t = []
    for j in range(n):
        expr = 0
        for k in range(n):
            expr += vec1[k]*vec2[(j+k) % n]
        t.append(expr)
    return np.array(t)


def get_edge_similarity(edge_label1, edge_label2):
    edge_label1_vec = get_edge_wordvector(edge_label1)
    edge_label2_vec = get_edge_wordvector(edge_label2)
    if edge_label1_vec is None or edge_label2_vec is None:
        if debug:
            print 'either of wordvectors is None'
        cs_words = -1
    else:
        cs_words = get_cosine_similarity(edge_label1, edge_label2, edge_label1_vec, edge_label2_vec)
        if not ((-1-1e-2) <= cs_words <= (1+1e-2)):
            print 'cs_words is ', cs_words
            raise AssertionError
    k = get_cosine_similarity_kernel(cs_words, -1)
    if debug:
        print '({}, {}, {})'.format(edge_label1, edge_label2, k)
    return k


def is_root_nodes_mismatch(amr1, amr2, cosine_threshold=None):
    def get_word_vec_local(wordvec, word_str):
        if wordvec is not None:
            return wordvec
        if not concept_regexp.sub('', word_str):
            word_str = concept_num_regexp.sub('', word_str)
        return get_wordvector(word_str)

    def get_joint_wordvec_frm_name_type(name_word_vec, type_word_vec):
        raise DeprecationWarning
        joint_wordvec = None
        if name_word_vec is not None:
            joint_wordvec = np.copy(name_word_vec)
        if type_word_vec is not None:
            if joint_wordvec is None:
                joint_wordvec = np.copy(type_word_vec)
            else:
                joint_wordvec = ss.fftconvolve(joint_wordvec, type_word_vec, mode='same')
        return joint_wordvec

    if debug:
        print '************************************'
        print 'evaluating similarity between nodes: '
        print '************************************'
        print 'amr1', amr1
        print 'amr2', amr2
    if is_label_kernel:
        if amr1.dummy_label == amr2.dummy_label:
            t_labels = 1
        else:
            if is_word_vectors_edge:
                t_labels = get_edge_similarity(amr1.dummy_label, amr2.dummy_label)
            else:
                t_labels = 0
        #
        if hasattr(amr1, 'dummy_label_inv') and hasattr(amr2, 'dummy_label_inv'):
            if amr1.dummy_label_inv == amr2.dummy_label_inv:
                t_labels += 1
            else:
                if is_word_vectors_edge:
                    t_labels += get_edge_similarity(amr1.dummy_label_inv, amr2.dummy_label_inv)
    if is_role_kernel:
        if hasattr(amr1, 'role') and hasattr(amr2, 'role'):
            if amr1.role != amr2.role:
                t_roles = 0
            else:
                t_roles = 1
        else:
            assert not (hasattr(amr1, 'role') or hasattr(amr2, 'role'))
            t_roles = 0
    if both_name_type:
        raise NotImplementedError
    elif is_name_else_type and amr1.name.lower() == amr2.name.lower():
        cs_words = 1
    elif (not is_name_else_type) and amr1.type.lower() == amr2.type.lower():
        cs_words = 1
    else:
        if is_word_vectors:
            if both_name_type:
                raise NotImplementedError
                amr1_wordvec = get_joint_wordvec_frm_name_type(amr1.name_wordvec, amr1.type_wordvec)
                amr2_wordvec = get_joint_wordvec_frm_name_type(amr2.name_wordvec, amr2.type_wordvec)
                amr1_word = amr1.name + ' ' + amr1.type
                amr2_word = amr2.name + ' ' + amr2.type
            else:
                #todo: use word vectors for both labels and names using semantic compositionality operators (say cross product)
                if is_name_else_type:
                    if get_word_vec_local(amr1.name_wordvec, amr1.name) is not None:
                        amr1_wordvec = get_word_vec_local(amr1.name_wordvec, amr1.name)
                        amr1_word = amr1.name
                    else:
                        amr1_wordvec = get_word_vec_local(amr1.type_wordvec, amr1.type)
                        amr1_word = amr1.type
                    if get_word_vec_local(amr2.name_wordvec, amr2.name) is not None:
                        amr2_wordvec = get_word_vec_local(amr2.name_wordvec, amr2.name)
                        amr2_word = amr2.name
                    else:
                        amr2_wordvec = get_word_vec_local(amr2.type_wordvec, amr2.type)
                        amr2_word = amr2.type
                else:
                    if debug:
                        print 'evaluating word vector similarity on types'
                    #
                    amr1_wordvec = get_word_vec_local(amr1.type_wordvec, amr1.type)
                    amr1_word = amr1.type
                    #
                    amr2_wordvec = get_word_vec_local(amr2.type_wordvec, amr2.type)
                    amr2_word = amr2.type
            if amr1_wordvec is None or amr2_wordvec is None:
                if debug:
                    print 'either of wordvectors is None'
                cs_words = 0
            else:
                cs_words = get_cosine_similarity(amr1_word, amr2_word, amr1_wordvec, amr2_wordvec)
                if not ((-1-1e-2) <= cs_words <= (1+1e-2)):
                    print 'cs_words is ', cs_words
                    raise AssertionError
        else:
            cs_words = 0
    if is_word_vectors:
        t_words = get_cosine_similarity_kernel(cs_words, cosine_threshold)
    else:
        t_words = cs_words
    if is_label_kernel and is_role_kernel:
        #todo: tune the parameter t_words_pow_fr_label_kernel
        t_words = t_words**(t_words_pow_fr_label_kernel*t_words_pow_fr_role_kernel)
        t = t_words*(t_words+t_labels+t_roles)
    elif is_label_kernel:
        #todo: tune the parameter t_words_pow_fr_label_kernel
        t_words = t_words**t_words_pow_fr_label_kernel
        t = t_words*(t_words+t_labels)
    else:
        t = t_words
    if t == 0:
        is_mismatch = True
    else:
        is_mismatch = False
    if debug:
        print 'is_mismatch', is_mismatch
        print 't', t
        print '************************************'
    return is_mismatch, t


def null_org(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    if debug:
        print 'u', u
        print 's', s
        print 'vh', vh
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T


def null(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    if debug:
        print 'u', u
        print 's', s
        print 'vh', vh
    if abs(s.min()) > eps:
        return np.array([])
    else:
        return s.min()*vh[s.argmin()]


def graph_kernel(amr1_nodes, amr2_nodes, lam=None, cosine_threshold=None, is_root_kernel=None, kernel_computations_map=kernel_computations_map_default):
    #amr1_nodes, amr2_nodes
    #if original kernel is true, first node should be root node in nodes list for the corresponding subgraphs
    #else the nodes should not be list be root node

    def core_eval_func(amr1, amr2):
        #amr1 and amr2 are root node of the corresponding AMRs
        #lam is lambda parameter
        if is_neighbor_kernel:
            raise AssertionError
        tuple_key_local = (amr1, amr2)
        root_nodes_mismatch, k_p = is_root_nodes_mismatch(amr1, amr2, cosine_threshold)
        if debug:
            print 'k_p is ', k_p
        if root_nodes_mismatch:
            return k_p #no need to save computations for such case
        else:
            #extend children list with other parents of the children
            if not is_inverse_centralize_amr:
                amr1_c = amr1.create_children_and_in_laws_list()
                amr2_c = amr2.create_children_and_in_laws_list()
            else:
                amr1_c = amr1.create_children_list()
                amr2_c = amr2.create_children_list()
            m = len(amr1_c)
            n = len(amr2_c)
            #todo: verify the case for min(m,n) = 0 is correct in the algorithm
            #if one of parents do not have any children, then similarity is zero
            #if both parents have no children, then also similarity is zero since we sum over matching sub-sequences
            if min(m, n) == 0:
                k_c = 0
            else:
                C = -1*np.ones((m, n)) #negative value represented unassigned value
                L = -1*np.ones((m, n)) #negative value represented unassigned value
                for i in range(m):
                    for j in range(n):
                        C[i, j] = evaluate_Cij(lam, cosine_threshold, C, L, amr1_c, amr2_c, i, j, is_root_kernel, kernel_computations_map)
                if debug:
                    print 'C', C
                k_c = C.sum()
            if debug:
                print 'k_p', k_p
                print 'k_c', k_c
            k_local = k_p + k_c
            #todo: temporarily added for debugging
            if k_c > 0.1*k_p and debug:
                print 'vvvvvvvvvvvvvvvvvvvvvvvvvvvvv'
                print 'k_p', k_p
                print 'k_c', k_c
                print '*********************************ratio', (k_c/k_p)
            return k_local

    if is_root_kernel:
        tuple_key = (amr1_nodes, amr2_nodes)
    else:
        tuple_key = (amr1_nodes[0], amr2_nodes[0])
    #
    if kernel_computations_map is not None:
        if tuple_key in kernel_computations_map:
            if debug:
                print '*********************************************************************************************'
            return kernel_computations_map[tuple_key]
    #
    if is_root_kernel:
        if debug:
            print 'evaluating original kernel between :'
            print amr1_nodes
            print amr2_nodes
        k = core_eval_func(amr1_nodes, amr2_nodes)
    else:
        k = core_eval_func(amr1_nodes[0], amr2_nodes[0])
        #note:this filtering of sparsity must be outside the core kernel evaluation function
        #here, before filtering relevant sparse nodes, we need to ensure that we do compare root node against root node also which correspond to standard kernel function and necessary for the analytical recursion
        for amr1 in filter_nodes_fr_sparse_cmp(amr1_nodes):
            for amr2 in filter_nodes_fr_sparse_cmp(amr2_nodes):
                if (amr1, amr2) != (amr1_nodes[0], amr2_nodes[0]):
                    k_local = core_eval_func(amr1, amr2)
                    k += k_local
    if k < 0:
        raise AssertionError
    if kernel_computations_map is not None:
        kernel_computations_map[tuple_key] = k
    if debug:
        print 'k', k
    return k


def graph_kernel_infinite(amr1_nodes, amr2_nodes, lam=None, cosine_threshold=None, is_intermediate_computation=False, coeff_recursive_calls=None):
    #amr1 and amr2 are root node of the corresponding AMRs
    def core_eval_func(amr1, amr2, local_coeff_recursive_calls):
        if local_coeff_recursive_calls is None:
            raise AssertionError
        tuple_key = (amr1, amr2)
        root_nodes_mismatch, k_p = is_root_nodes_mismatch(amr1, amr2, cosine_threshold)
        print 'k_p is ', k_p
        if root_nodes_mismatch:
            local_coeff_recursive_calls[tuple_key]['zero_order'] = k_p
            return False
        else:
            local_coeff_recursive_calls[tuple_key]['zero_order'] = k_p
            amr1_c = amr1.create_undirected_children_list()
            amr2_c = amr2.create_undirected_children_list()
            m = len(amr1_c)
            n = len(amr2_c)
            L = -1*np.ones((m, n)) #negative value represented unassigned value
            for i in range(m):
                for j in range(n):
                    evaluate_Cij_infinite(lam, cosine_threshold, L, amr1_c, amr2_c, i, j, local_coeff_recursive_calls, parent1=amr1, parent2=amr2, expr_coeff=1)
            if debug:
                print 'k_p', k_p
            return True

    if is_intermediate_computation:
        if coeff_recursive_calls is None:
            raise AssertionError
    tuple_key = (amr1_nodes, amr2_nodes)
    if coeff_recursive_calls is None:
        coeff_recursive_calls = {}
    coeff_recursive_calls[tuple_key] = {}
    if debug:
        print 'evaluating original kernel between :'
        print amr1_nodes
        print amr2_nodes
    result_boolean = core_eval_func(amr1_nodes, amr2_nodes, coeff_recursive_calls)
    if not result_boolean:
        k = 0
    else:
        if not is_intermediate_computation: #it means that all intermediate computations are also completed
            #solve the set of linear equations
            # both methods should return all calls
            all_computations_keys = coeff_recursive_calls.keys()
            for curr_computation in coeff_recursive_calls:
                all_computations_keys += coeff_recursive_calls[curr_computation].keys()
            all_computations_keys = my_util.unique_list(all_computations_keys)
            all_computations_keys.remove('zero_order')
            num_computations = len(all_computations_keys)
            N = np.zeros(num_computations)
            M = np.zeros((num_computations, num_computations))
            for i in range(num_computations):
                curr_computation = all_computations_keys[i]
                for j in range(num_computations):
                    curr_dep_computation = all_computations_keys[j]
                    if curr_dep_computation in coeff_recursive_calls[curr_computation]:
                        M[i, j] = coeff_recursive_calls[curr_computation][curr_dep_computation]
                N[i] = -coeff_recursive_calls[curr_computation]['zero_order']
                #do some assertions here
                for stored_key in coeff_recursive_calls[curr_computation]:
                    if stored_key not in all_computations_keys and stored_key != 'zero_order':
                        raise AssertionError
            if np.all(N == 0):
                raise AssertionError
            A = M-np.eye(num_computations)
            if debug:
                print 'M', M
                print 'A', A
                print 'N', N
            idx_non_intermediate_kernel = all_computations_keys.index(tuple_key)
            det_A = np.linalg.det(A)
            if det_A <= 0:
                print 'determinant is not positive .....'
                print 'logging details for debug ...'
                print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
                print 'det of A is ', det_A
                print 'eigens of A are ', np.linalg.eigvals(A)
                print 'rank of A is ', np.linalg.matrix_rank(A)
                sols = np.linalg.solve(A, N)
                print 'sols', sols
                print 'sols[idx_non_intermediate_kernel] is ', sols[idx_non_intermediate_kernel]
                print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
                print 'see if solutions with pseudo inverse can work'
                sols_pi = np.dot(np.linalg.pinv(A), N)
                print 'sols_pi ', sols_pi
                print 'sols_pi[idx_non_intermediate_kernel] is ', sols_pi[idx_non_intermediate_kernel]
                raise AssertionError
            sols = np.linalg.solve(A, N)
            if debug:
                print 'sols', sols
                for i in range(num_computations):
                    print '({},{})'.format(all_computations_keys[i][0].name, all_computations_keys[i][1].name)
                    print sols[i]
                print 'sols[idx_non_intermediate_kernel] is ', sols[idx_non_intermediate_kernel]
            # if not np.all(sols >= 0):
            #     raise AssertionError
            if sols[idx_non_intermediate_kernel] < 0:
                raise AssertionError
                # print 'invalid kernel value'
            k = sols[idx_non_intermediate_kernel]
    if not is_intermediate_computation:
        return k
    else:
        return


def evaluate_Cij_infinite(lam, cosine_threshold, L, amr1_c, amr2_c, i, j, coeff_recursive_calls, parent1, parent2, expr_coeff):
    if None in [coeff_recursive_calls, parent1, parent2, expr_coeff]:
        raise AssertionError
    (m, n) = L.shape
    if i >= m or i < 0:
            return
    if j >= n or j < 0:
            return
    tuple_key = (amr1_c[i], amr2_c[j])
    if debug:
        print 'amr1_c[i].name', amr1_c[i].name
        print 'amr2_c[j].name', amr2_c[j].name
    root_nodes_mismatch, t = is_root_nodes_mismatch(amr1_c[i], amr2_c[j], cosine_threshold)
    if root_nodes_mismatch:
        if debug:
            print 'root nodes mismatch '
    else:
        expr = (float(1 - (lam**evaluate_Lij(L, i, j)))*lam)/(1-lam)
        expr *= t
        parent_tuple = (parent1, parent2)
        if parent_tuple not in coeff_recursive_calls:
            coeff_recursive_calls[parent_tuple] = {}
        if tuple_key not in coeff_recursive_calls[parent_tuple]:
            coeff_recursive_calls[parent_tuple][tuple_key] = expr*expr_coeff
        else:
            coeff_recursive_calls[parent_tuple][tuple_key] += expr*expr_coeff
        if tuple_key not in coeff_recursive_calls:
            graph_kernel_infinite(amr1_c[i], amr2_c[j], lam, cosine_threshold, is_intermediate_computation=True, coeff_recursive_calls=coeff_recursive_calls)
        evaluate_Cij_infinite(lam, cosine_threshold, L, amr1_c, amr2_c, i+1, j+1, coeff_recursive_calls, parent1=parent1, parent2=parent2, expr_coeff=(expr_coeff*lam))


def filter_nodes_fr_sparse_cmp(nodes):
    if not is_sparse:
        return nodes
    #todo: see if this is enough for identifying a concept node or do we also need to include non-concept nodes while ignoring named entities
    new_nodes_list = []
    for node in nodes:
        #either node should be a concept node with at least one child
        if not is_neighbor_kernel:
            if (node.name == node.type or node.type is None):
                if (not is_inverse_centralize_amr) and len(node.create_children_and_in_laws_list()) >= min_children_sparse:
                    new_nodes_list.append(node)
                elif is_inverse_centralize_amr and len(node.create_children_list()) >= min_children_sparse:
                    new_nodes_list.append(node)
        else:
            if (node.name == node.type or node.type is None) and len(node.create_undirected_children_list()) >= min_children_sparse:
                    new_nodes_list.append(node)
    return new_nodes_list


def get_updated_kernel_calls_stack(stack_map, val):
    stack_map = copy.copy(stack_map)
    for key in stack_map:
        stack_map[key] *= val
    return stack_map


def update_coeff_recursive_calls(coeff_recursive_calls, kernel_calls_stack, key):
    for curr_key in kernel_calls_stack:
        if key not in coeff_recursive_calls[curr_key]:
            coeff_recursive_calls[curr_key][key] = kernel_calls_stack[curr_key]
        else:
            coeff_recursive_calls[curr_key][key] += kernel_calls_stack[curr_key]


def evaluate_Cij(lam, cosine_threshold, C, L, amr1_c, amr2_c, i, j, is_root_kernel, kernel_computations_map):
    (m, n) = L.shape
    if i >= m or i < 0:
        return 0
    if j >= n or j < 0:
        return 0
    if C[i, j] < 0: #value already assigned if not negative
        if debug:
            print 'amr1_c[i].name', amr1_c[i].name
            print 'amr2_c[j].name', amr2_c[j].name
        root_nodes_mismatch, t = is_root_nodes_mismatch(amr1_c[i], amr2_c[j], cosine_threshold)
        if root_nodes_mismatch:
            if debug:
                print 'root nodes mismatch '
            C[i, j] = 0
        else:
            expr = (float(1 - (lam**evaluate_Lij(L, i, j)))*lam)/(1-lam)
            if is_word_vectors:
                expr *= t
            if is_root_kernel:
                expr *= graph_kernel(amr1_c[i], amr2_c[j], lam, cosine_threshold, is_root_kernel, kernel_computations_map)
            else:
                if not is_inverse_centralize_amr:
                    expr *= graph_kernel(amr1_c[i].create_descendants_inc_inlaws_list(), amr2_c[j].create_descendants_inc_inlaws_list(), lam, cosine_threshold, is_root_kernel, kernel_computations_map)
                else:
                    if hasattr(amr1_c[i], 'create_descendants_list'):
                        amr1_c_i_descendants_list = amr1_c[i].create_descendants_list()
                        amr2_c_j_descendants_list = amr2_c[j].create_descendants_list()
                    else:
                        amr1_c_i_descendants_list = ead.get_descendants(amr1_c[i])
                        amr2_c_j_descendants_list = ead.get_descendants(amr2_c[j])
                    expr *= graph_kernel(amr1_c_i_descendants_list, amr2_c_j_descendants_list, lam, cosine_threshold, is_root_kernel, kernel_computations_map)
            Cij_p11 = evaluate_Cij(lam, cosine_threshold, C, L, amr1_c, amr2_c, i+1, j+1, is_root_kernel, kernel_computations_map)
            expr += lam*Cij_p11
            C[i, j] = expr
    return C[i, j]


def evaluate_Lij(L, i, j):
    (m, n) = L.shape
    if i >= m or i < 0:
        return 0
    if j >= n or j < 0:
        return 0
    if L[i, j] < 0:
        L[i, j] = evaluate_Lij(L, i+1, j+1)+1
    return L[i, j]


def get_nodes_list_with_root_zero_idx(amr_map):
        amr_nodes = amr_map.values()
        amr_nodes.insert(0, amr_map[root])
        amr_nodes = my_util.unique_list(amr_nodes)
        return amr_nodes


def graph_kernel_wrapper(nodes1, nodes2, lam=None, cosine_threshold=None, is_root_kernel=None):
    if is_root_kernel:
        if debug:
            print 'nodes1[root] ', nodes1[root]
            print 'nodes2[root] ', nodes2[root]
        if is_neighbor_kernel:
            if is_parent_child_coeff:
                k = graph_kernel_infinite(amr1_nodes=nodes1[root], amr2_nodes=nodes2[root], lam=lam, cosine_threshold=cosine_threshold)
            else:
                raise NotImplementedError
        else:
            k = graph_kernel(
                amr1_nodes=nodes1[root],
                amr2_nodes=nodes2[root],
                lam=lam,
                cosine_threshold=cosine_threshold,
                is_root_kernel=is_root_kernel)
    else:
        if is_neighbor_kernel:
            raise NotImplementedError
        else:
            kernel_computations_map = {}
            k = graph_kernel(
                amr1_nodes=get_nodes_list_with_root_zero_idx(nodes1),
                amr2_nodes=get_nodes_list_with_root_zero_idx(nodes2),
                lam=lam,
                cosine_threshold=cosine_threshold,
                is_root_kernel=is_root_kernel,
                kernel_computations_map=kernel_computations_map)
    if debug:
        print 'kernel similarity is ', k
    return k


def get_amrs(amr_dot_file, start_amr, end_amr):
    amr_dot_files = []
    for i in range(start_amr, end_amr+1):
        curr_amr_dot_file = amr_dot_file + '.' + str(i) + '.dot'
        if debug:
            print 'curr_amr_dot_file:', curr_amr_dot_file
        amr_dot_files.append(curr_amr_dot_file)
    return amr_dot_files


def get_default_amrs():
    amr_dot_files = []
    labels = []
    for curr_amr in default_amrs:
        amr_dot_files += get_amrs(curr_amr['path'], curr_amr['idx'][0], curr_amr['idx'][1])
        labels += curr_amr['labels']
    labels = np.array(labels)
    count = 0
    for curr_amr_dot_file in amr_dot_files:
        print str(count) + ' : ' + curr_amr_dot_file + ' : ' + str(labels[count])
        count += 1
    #representing invalids as ones and vice versa
    labels = 1-labels
    # #representing the labels as strings
    # labels = labels.astype(np.str).astype(np.object)
    return amr_dot_files, labels


def get_amr_graphs_frm_dot(amr_dot_files):
    print 'getting amr graphs from dot files ...'
    start_time = time.time()
    n = len(amr_dot_files)
    amr_graphs = np.empty(dtype=np.object, shape=(n, 1))
    for i in range(n):
        amr_dot_file = amr_dot_files[i]
        amr_graphs[i, 0] = {'path': amr_dot_file, 'nodes': ead.build_nodes_tree_from_amr_dot_file_and_simplify(amr_dot_file)[0]}
    print 'Execution time to get graphs from dot files was ', time.time() - start_time
    if is_word_vectors:
        print 'Preprocessing AMR graphs for assigning word vectors ...'
        start_time = time.time()
        for i in range(n):
            preprocess_amr_fr_assign_wordvector(amr_graphs[i, 0])
        print 'Execution time to assign vectors for each node in AMR graphs was ', time.time()-start_time
    return amr_graphs


def preprocess_amr_fr_assign_wordvector(amr_graph):
    def get_wordvec_local(word):
        if not concept_regexp.sub('', word):
            word = concept_num_regexp.sub('', word)
            # print word
        return get_wordvector(word)

    for node in amr_graph['nodes'].values():
        if both_name_type:
            #name
            node.name_wordvec = get_wordvec_local(node.name)
            #type
            node.type_wordvec = get_wordvec_local(node.type)
        else:
            if is_name_else_type:
                # name
                node.name_wordvec = get_wordvec_local(node.name)
            else:
                #type
                node.type_wordvec = get_wordvec_local(node.type)


def eval_graph_kernel_matrix(amr_graphs1, amr_graphs2, lam=None, cosine_threshold=None, is_root_kernel=is_root_kernel_default):
    n1 = amr_graphs1.shape[0]
    n2 = amr_graphs2.shape[0]
    # if n1 != n2:
    #     raise NotImplementedError
    start_time = time.time()
    K = -float('Inf')*np.ones((n1, n2)) #negative represents unassigned similarity
    # if not self_fr_test:
    is_error=False
    num_kernel_errors = 0
    #
    if is_normalize_each_kernel:
        self_k_i = -float('Inf')*np.ones(n1)
        self_k_j = -float('Inf')*np.ones(n2)
    #
    for i in range(n1):
        for j in range(n2):
            if K[i, j] == -float('Inf'):
                if coarse_debug:
                    print 'amr_dot_files[i]', amr_graphs1[i, 0]['path']
                    print 'amr_dot_files[j]', amr_graphs2[j, 0]['path']
                if not is_normalize_each_kernel:
                    raise AssertionError
                try:
                    # todo: self similarity computations are beings repeated for each pair. This should be avoided and should save computational cost significantly.
                    # todo: also deep copy operations may be very expensive. See if it can be avoided some how. Anyways, it would be reduced if we reduce
                    # todo: duplicate self similarity computations
                    start_time_local = time.time()
                    Kij = graph_kernel_wrapper(nodes1=amr_graphs1[i, 0]['nodes'], nodes2=amr_graphs2[j, 0]['nodes'], lam=lam, cosine_threshold=cosine_threshold, is_root_kernel=is_root_kernel)
                    if debug:
                        print 'Kij ', Kij
                    if is_normalize_each_kernel:
                        # assuming that computations are saved in a global map so that we don't have to bother about same calls from here
                        if self_k_i[i] == -float('Inf'):
                            Kii = graph_kernel_wrapper(nodes1=amr_graphs1[i, 0]['nodes'], nodes2=copy.deepcopy(amr_graphs1[i, 0]['nodes']), lam=lam, cosine_threshold=cosine_threshold, is_root_kernel=is_root_kernel)
                            self_k_i[i] = Kii
                        else:
                            Kii = self_k_i[i]
                        if debug:
                            print 'Kii ', Kii
                        #
                        if self_k_j[j] == -float('Inf'):
                            Kjj = graph_kernel_wrapper(nodes1=amr_graphs2[j, 0]['nodes'], nodes2=copy.deepcopy(amr_graphs2[j, 0]['nodes']), lam=lam, cosine_threshold=cosine_threshold, is_root_kernel=is_root_kernel)
                            self_k_j[j] = Kjj
                        else:
                            Kjj = self_k_j[j]
                        if debug:
                            print 'Kjj ', Kjj
                        #
                        K[i, j] = Kij/math.sqrt(Kii*Kjj)
                        if debug:
                            print 'normalized K[i, j] is ', K[i, j]
                        if not ((0-1e-2) <= K[i,j] <= (1+1e-2)):
                            print '******************************************'
                            print 'unexpected value of K[i,j] ', K[i,j]
                            print 'Kij is ', Kij
                            print 'Kii is ', Kii
                            print 'Kjj is ', Kjj
                            print 'amr_dot_files[i]', amr_graphs1[i, 0]['path']
                            print 'amr_dot_files[j]', amr_graphs2[j, 0]['path']
                            print '******************************************'
                            raise AssertionError
                    if coarse_debug:
                        print 'time to computer current kernel was ', time.time()-start_time_local
                except:
                        print 'error computing kernel between '
                        print 'amr_dot_files[i]', amr_graphs1[i, 0]['path']
                        print 'amr_dot_files[j]', amr_graphs2[j, 0]['path']
                        is_error = True
                        num_kernel_errors += 1
                        raise
                if debug:
                    print 'K[i, j] ', K[i, j]
    if is_error:
        print 'Number of kernel computation errors ', num_kernel_errors
        raise AssertionError
    if debug:
        print 'Kernel Matrix is ', K
    #todo: normalize the matrix
    K_un_norm = np.copy(K)
    if not is_normalize_each_kernel:
        # normalization should not be at matrix level (fundamentally wrong if the two arrays of graphs amr_graphs1, amr_graphs2 are different sources)
        raise DeprecationWarning
        if n1 == n2:
            norm = np.sqrt(K.diagonal().repeat(n1).reshape((n1, n1)))
            K /= norm*norm.transpose()
            print 'Kernel Matrix after normalization is ', K
        else:
            raise NotImplementedError
    if is_laplace_kernel: #this seems to give very bad results though
        raise DeprecationWarning
        K = get_laplace_transform(K)
    if is_nonlinear_kernel:
        raise DeprecationWarning
        K = nonlinear_func_on_kernel_matrix(K)
    if is_reg_kernel:
        K += reg_lambda*np.eye(n1)
    if is_pos_def_test:
        if not is_semi_pos_def(K):
            raise AssertionError
    #todo: also write normalization code for other cases on relationship between n1 and n2
    print 'Time to compute the kernel matrix is ', time.time()-start_time
    matrix_name = './gen_kernel_matrices/'
    if is_sparse:
        matrix_name += 'graph_kernel_sparse_K_'
    else:
        matrix_name += 'graph_kernel_K_'
    matrix_name += str(n1)+'_'+str(n2)+'_lambda'+str(lam)
    if is_word_vectors:
        matrix_name += '_ct'+str(cosine_threshold)
    if is_save_matrix:
        if not ch.is_hpcc:
            start_time = time.time()
            np.save(cap.absolute_path+matrix_name, K)
            np.save(cap.absolute_path+matrix_name+'_unnorm', K_un_norm)
            print 'Time to save the matrix was ', time.time()-start_time
    if is_save_matrix_img:
        if not ch.is_hpcc:
            start_time = time.time()
            #plotting
            plt.pcolor(K)
            plt.colorbar()
            plt.savefig(cap.absolute_path+matrix_name+'.pdf', dpi=80)
            plt.close()
            print 'Time to save the matrix image was ', time.time()-start_time
    return K


def get_laplace_transform(K):
    #assuming a square matrix
    n = K.shape[0]
    K_ = np.copy(K)
    K = None
    K_[range(n), range(n)] = 0
    G = np.diag(K_.sum(0))
    L = G - K_
    return L


def is_semi_pos_def(K):
    eig_val = np.linalg.eigvals(K)
    if np.all(eig_val >= -1e-2):
        return True
    else:
        print 'determinant (product of eigens) is ', eig_val.prod()
        print 'trace (sum of eigens) is ', eig_val.sum()
        print 'eigens', eig_val
        return False
    #
    #
    # try:
    #     np.linalg.cholesky(K)
    #     is_pos_def_mat = True
    # except:
    #     is_pos_def_mat = False
    # return is_pos_def_mat
    #


def nonlinear_func_on_kernel_matrix(K):
    u, s, v = np.linalg.svd(K)
    if nonlinear_func == 'log':
        s = np.log(s+1)
    elif nonlinear_func == 'pow':
        s = np.sign(s)*(np.abs(s)**nonlinear_pow)
    K = np.dot(np.dot(u, np.diag(s)), v)
    print 'Kernel Matrix after nonlinear transform is ', K
    return K


def tune_svm_kernel_cosine_threshold(amr_graphs, labels, min_ct, max_ct, score_min_ct=None, score_max_ct=None, opt_lam_min_ct=None, opt_lam_max_ct=None):
    # todo: also implement 1-D grid search for each variable
    # todo: apply binary search within the finally optimized grid
    if score_min_ct is None:
        opt_lam_min_ct, score_min_ct = tune_svm_kernel_lambda(amr_graphs, labels, lambda_range_min, lambda_range_max, cosine_threshold=min_ct)
    if score_max_ct is None:
        opt_lam_max_ct, score_max_ct = tune_svm_kernel_lambda(amr_graphs, labels, lambda_range_min, lambda_range_max, cosine_threshold=max_ct)
    ct_diff = max_ct - min_ct
    if ct_diff < 0:
        raise AssertionError
    elif ct_diff < ct_tol:
        if score_min_ct['mean'] > score_max_ct['mean']:
            return min_ct, opt_lam_min_ct, score_min_ct
        else:
            return max_ct, opt_lam_max_ct, score_max_ct
    else:
        ct_tuning_branch = 2
        if score_min_ct['mean'] > score_max_ct['mean']:
            mean_ct = (float(ct_diff)/ct_tuning_branch) + min_ct
            return tune_svm_kernel_cosine_threshold(amr_graphs, labels, min_ct, mean_ct, score_min_ct=score_min_ct, opt_lam_min_ct=opt_lam_min_ct)
        else:
            mean_ct = max_ct - (float(ct_diff)/ct_tuning_branch)
            return tune_svm_kernel_cosine_threshold(amr_graphs, labels, mean_ct, max_ct, score_max_ct=score_max_ct, opt_lam_max_ct=opt_lam_max_ct)


def tune_svm_kernel_lambda(amr_graphs, labels, min_lam, max_lam, score_min_lam=None, score_max_lam=None, cosine_threshold=None):
    # todo: also implement 1-D grid search for each variable
    # todo: apply binary search within the finally optimized grid
    if score_min_lam is None:
        score_min_lam = train_svm_kernel(amr_graphs, labels, min_lam, cosine_threshold)
    if score_max_lam is None:
        score_max_lam = train_svm_kernel(amr_graphs, labels, max_lam, cosine_threshold)
    lam_diff = max_lam-min_lam
    if lam_diff < 0:
        raise AssertionError
    elif lam_diff < lambda_tol:
        if score_min_lam['mean'] > score_max_lam['mean']:
            return min_lam, score_min_lam
        else:
            return max_lam, score_max_lam
    else:
        if lambda_tuning_branch < 2:
            raise AssertionError
        if score_min_lam['mean'] > score_max_lam['mean']:
            mean_lam = (float(lam_diff)/lambda_tuning_branch) + min_lam
            return tune_svm_kernel_lambda(amr_graphs, labels, min_lam, mean_lam, score_min_lam=score_min_lam, cosine_threshold=cosine_threshold)
        else: #breaking tie in favour of max lambda
            mean_lam = max_lam - (float(lam_diff)/lambda_tuning_branch)
            return tune_svm_kernel_lambda(amr_graphs, labels, mean_lam, max_lam, score_max_lam=score_max_lam, cosine_threshold=cosine_threshold)


def train_svm_kernel(amr_graphs, labels, lam, cosine_threshold=None):
    print 'lambda: ', lam
    print 'cosine_threshold: ', cosine_threshold
    n = amr_graphs.shape[0]
    K = eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=lam, cosine_threshold=cosine_threshold, is_root_kernel=is_root_kernel_default)
    num_trials = num_trials_svm
    adj_rand_score_test = -1*np.ones(num_trials)
    #max
    max_adj_rand_score = -1
    max_ars_train_labels = None
    max_ars_test_labels = None
    max_ars_test_labels_pred = None
    max_ars_train_idx = None
    max_ars_test_idx = None
    max_ars_precision = None
    max_ars_recall = None
    max_ars_f1 = None
    max_ars_auc = None
    max_ars_zero_one_loss = None
    max_ars_jss = None
    #mean
    mean_ars_adjusted_rand_score = None
    mean_ars_train_labels = None
    mean_ars_test_labels = None
    mean_ars_test_labels_pred = None
    mean_ars_train_idx = None
    mean_ars_test_idx = None
    mean_ars_precision = None
    mean_ars_recall = None
    mean_ars_f1 = None
    mean_ars_auc = None
    mean_ars_zero_one_loss = None
    mean_ars_jss = None
    print 'Learning SVM classifier ...'
    # print 'lambda is ', lam
    start_time = time.time()
    for i in range(num_trials):
        num_train_samples = round(K.shape[0]*training_frac)
        # train = rnd.randint(0, n-1, (num_train_samples,))
        all_idx = np.arange(n)
        np.random.shuffle(all_idx)
        train = all_idx[:num_train_samples]
        test = np.setdiff1d(np.arange(0, n), train)
        K_train = K[np.meshgrid(train, train)]
        K_test = K[np.meshgrid(train, test)]
        labels_train = labels[train]
        labels_test = labels[test]
        # num_ones = labels_train.sum()
        # weight_0 = (num_ones/float(num_train_samples))**1
        # weight_1 = 1-weight_0
        # print 'weight_1', weight_1
        # print 'weight_0', weight_0
        # sample_weights_train = weight_1*labels_train + weight_0*(1-labels_train)
        svm_clf = skl_svm.SVC(kernel='precomputed', probability=True, verbose=is_svm_verbose, class_weight='auto')
        svm_clf.fit(K_train, labels_train) #, sample_weight=sample_weights_train)
        # print 'Inferring labels for the test data ...'
        # start_time = time.time()
        labels_test_pred = svm_clf.predict(K_test)
        # print 'Test labels are: ', labels_test
        # print 'Predicted labels are: ', labels_test_pred
        # print 'Inference time was ', time.time()-start_time
        precision = skm.precision_score(labels_test.astype(np.int), labels_test_pred.astype(np.int))
        # print 'Precision: ', precision
        recall = skm.recall_score(labels_test.astype(np.int), labels_test_pred.astype(np.int))
        # print 'Recall: ', recall
        f1 = skm.f1_score(labels_test.astype(np.int), labels_test_pred.astype(np.int))
        # print 'F1: ', f1
        # auc = skm.roc_auc_score(labels_test.astype(np.int), labels_test_pred.astype(np.int))
        # # print 'AUC:', auc
        zero_one_loss = skm.zero_one_loss(labels_test.astype(np.int), labels_test_pred.astype(np.int))
        # print 'zero_one_loss: ', zero_one_loss
        jss = skm.jaccard_similarity_score(labels_test.astype(np.int), labels_test_pred.astype(np.int))
        # print 'jaccard_similarity_score: ', jss
        # print '****************************************************'
        adj_rand_score_test[i] = skm.adjusted_rand_score(labels_test.astype(np.int), labels_test_pred.astype(np.int))
        if debug:
            print 'adj_rand_score_test[i]', adj_rand_score_test[i]
        if adj_rand_score_test[i] > max_adj_rand_score:
            max_adj_rand_score = adj_rand_score_test[i]
            max_ars_train_labels = labels_train
            max_ars_test_labels = labels_test
            max_ars_test_labels_pred = labels_test_pred
            max_ars_train_idx = train
            max_ars_test_idx = test
            max_ars_precision = precision
            max_ars_recall = recall
            max_ars_f1 = f1
            # max_ars_auc = auc
            max_ars_zero_one_loss = zero_one_loss
            max_ars_jss = jss
        if abs(adj_rand_score_test.mean()-adj_rand_score_test[i]) < 1e-3:
            mean_ars_adjusted_rand_score = adj_rand_score_test[i]
            mean_ars_train_labels = labels_train
            mean_ars_test_labels = labels_test
            mean_ars_test_labels_pred = labels_test_pred
            mean_ars_train_idx = train
            mean_ars_test_idx = test
            mean_ars_precision = precision
            mean_ars_recall = recall
            mean_ars_f1 = f1
            # mean_ars_auc = auc
            mean_ars_zero_one_loss = zero_one_loss
            mean_ars_jss = jss
        # print 'adjusted rand score: ', adj_rand_score_test[i]
        # print '****************************************************'
        # print 'Inferring labels for training data itself ...'
        # start_time = time.time()
        # labels_train_pred = svm_clf.predict(K_train)
        # print 'Training labels are: ', labels_train
        # print 'Predicted training labels are: ', labels_train_pred
        # print 'Inference time was ', time.time()-start_time
        # print 'adjusted rand score: ', skm.adjusted_rand_score(labels_train.astype(np.int), labels_train_pred.astype(np.int))
        # print 'Precision: ', skm.precision_score(labels_train.astype(np.int), labels_train_pred.astype(np.int))
        # print 'Recall: ', skm.recall_score(labels_train.astype(np.int), labels_train_pred.astype(np.int))
        # print 'F1: ', skm.f1_score(labels_train.astype(np.int), labels_train_pred.astype(np.int))
    print 'Learning time was ', time.time()-start_time
    print '****************************************************'
    print '****************************************************'
    print 'Maximum adjusted rand score for test ', adj_rand_score_test.max()
    print 'Training labels for max ars ', max_ars_train_labels
    print 'Test labels for max ars ', max_ars_test_labels
    print 'Test labels inferred for max ars ', max_ars_test_labels_pred
    idx_train_pos = max_ars_train_idx[np.nonzero(max_ars_train_labels == 1)].tolist()
    idx_train_pos.sort()
    print 'Indices of training positives for max ars ', idx_train_pos
    print 'Indices of test positives for max ars ', max_ars_test_idx[np.nonzero(max_ars_test_labels == 1)]
    print 'Indices of inferred test positives for max ars ', max_ars_test_idx[np.nonzero(max_ars_test_labels_pred == 1)]
    print 'Training idx for max ars ', max_ars_train_idx
    print 'Test idx for max ars ', max_ars_test_idx
    print 'Precision for max ars ', max_ars_precision
    print 'Recall for max ars ', max_ars_recall
    print 'F1 for max ars ', max_ars_f1
    print 'AUC for max ars ', max_ars_auc
    print 'Zero one loss for max ars ', max_ars_zero_one_loss
    print 'Jaccard Similarity Score  for max ars ', max_ars_jss
    max_ars_cm = skm.confusion_matrix(max_ars_test_labels, max_ars_test_labels_pred)
    print 'confusion matrix for max ars ', max_ars_cm
    print '****************************************************'
    print 'Minimum adjusted rand score for test ', adj_rand_score_test.min()
    print 'Mean adjusted rand score for test ', adj_rand_score_test.mean()
    print 'SD adjusted rand score for test ', np.std(adj_rand_score_test)
    print '****************************************************'
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print 'adjusted rand score for mean ars ', mean_ars_adjusted_rand_score
    print 'Training labels for mean ars ', mean_ars_train_labels
    print 'Test labels for mean ars ', mean_ars_test_labels
    print 'Test labels inferred for mean ars ', mean_ars_test_labels_pred
    idx_train_pos = mean_ars_train_idx[np.nonzero(mean_ars_train_labels == 1)]
    idx_train_pos.sort()
    print 'Indices of training positives for mean ars ', idx_train_pos
    print 'Indices of test positives for mean ars ', mean_ars_test_idx[np.nonzero(mean_ars_test_labels == 1)].tolist()
    print 'Indices of inferred test positives for mean ars ', mean_ars_test_idx[np.nonzero(mean_ars_test_labels_pred == 1)]
    print 'Training idx for mean ars ', mean_ars_train_idx
    print 'Test idx for mean ars ', mean_ars_test_idx
    print 'Precision for mean ars ', mean_ars_precision
    print 'Recall for mean ars ', mean_ars_recall
    print 'F1 for mean ars ', mean_ars_f1
    print 'AUC for mean ars ', mean_ars_auc
    print 'Zero one loss for mean ars ', mean_ars_zero_one_loss
    print 'Jaccard Similarity Score for mean ars ', mean_ars_jss
    mean_ars_cm = skm.confusion_matrix(mean_ars_test_labels, mean_ars_test_labels_pred)
    print 'confusion matrix for mean ars ', mean_ars_cm
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++'
    adj_rand_score_result = {}
    adj_rand_score_result['mean'] = adj_rand_score_test.mean()
    adj_rand_score_result['max'] = adj_rand_score_test.max()
    adj_rand_score_result['min'] = adj_rand_score_test.min()
    adj_rand_score_result['std'] = np.std(adj_rand_score_test)
    return adj_rand_score_result


def graph_kernels_func(amr_dot_file, start_amr, end_amr, lam=None, cosine_threshold=None, is_root_kernel=is_root_kernel_default):
    if amr_dot_file is not None and amr_dot_file:
        amr_dot_files = get_amrs(amr_dot_file, start_amr, end_amr)
    else:
        amr_dot_files, labels = get_default_amrs()
        print 'labels are ', labels
    #getting graphs from dot files
    amr_graphs = get_amr_graphs_frm_dot(amr_dot_files)
    n = amr_graphs.shape[0] #assuming the matrix is a square
    if labels is not None and is_svm:
        if is_tuning:
            if is_word_vectors:
                opt_ct, opt_lam, opt_lam_ars = tune_svm_kernel_cosine_threshold(amr_graphs, labels, ct_range_min, ct_range_max)
                print 'Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is ', opt_ct
            else:
                opt_lam, opt_lam_ars = tune_svm_kernel_lambda(amr_graphs, labels, lambda_range_min, lambda_range_max)
            print 'Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is ', opt_lam
            print 'Mean adjusted rand score for optimal lambda is ', opt_lam_ars['mean']
            print 'Max adjusted rand score for optimal lambda is ', opt_lam_ars['max']
            print 'Min adjusted rand score for optimal lambda is ', opt_lam_ars['min']
            print 'SD adjusted rand score for optimal lambda is ', opt_lam_ars['std']
        else:
            if is_word_vectors:
                train_svm_kernel(amr_graphs, labels, lam, cosine_threshold)
            else:
                train_svm_kernel(amr_graphs, labels, lam)
    elif is_spectral_clustering:
        K = eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=lam, is_root_kernel=is_root_kernel)
        for i in range(5):
            print 'Learning labels with spectral clustering ...'
            start_time = time.time()
            num_clusters = 2
            spc_obj = skc.SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
            print K.shape
            labels_pred = spc_obj.fit_predict(K)
            print 'Time to predict labels with spectral clustering was ', time.time()-start_time
            print 'Predicted labels are: ', labels_pred
            print 'Index for zero labels are ', np.nonzero(labels_pred == 0)
            print 'Index for one labels are ', np.nonzero(labels_pred == 1)
            if num_clusters == 3:
                print 'Index for 2 labels are ', np.nonzero(labels_pred == 2)
            if labels is not None:
                print 'True label are: ', labels
                print 'adjusted rand score: ', skm.adjusted_rand_score(labels, labels_pred)
    elif is_corex:
        raise NotImplementedError
        K = eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=lam, is_root_kernel=is_root_kernel)
        num_trials = 10
        crxobjs = np.empty(shape=num_trials, dtype=np.object)
        tc = np.ones(num_trials)
        print 'Learning labels with CORrelation EXplanation clustering ...'
        start_time = time.time()
        is_single_hidden = True
        for i in range(num_trials):
            if is_single_hidden:
                crxobjs[i] = crx.Marginal_Corex(n_hidden=1, dim_hidden=2, tree=True, marginal_description='gaussian', max_iter=10000, eps=1e-10)
            else:
                crxobjs[i] = crx.Marginal_Corex(n_hidden=2, dim_hidden=1, tree=True, marginal_description='gaussian', max_iter=10000, eps=1e-10)
            crxobjs[i].fit(K)
            if is_single_hidden:
                labels_pred = crxobjs[i].labels.reshape(-1)
            else:
                labels_pred = crxobjs[i].clusters
            if is_single_hidden:
                tc[i] = crxobjs[i].tcs
            else:
                tc[i] = crxobjs[i].tcs.sum()
            # print 'Total correlation: ', tc[i]
            # print 'Predicted labels are: ', labels_pred
            # print 'Index for zero labels are ', np.nonzero(labels_pred == 0)
            # print 'Index for one labels are ', np.nonzero(labels_pred == 1)
            # if labels is not None:
            #     print 'True label are: ', labels
            #     print 'adjusted rand score: ', skm.adjusted_rand_score(labels, labels_pred)
        print 'Time to predict labels with corex was ', time.time()-start_time
        # print '******************** The most optimal *********************'
        print 'tc ', tc
        max_idx = tc.argmax()
        print 'Total correlation: ', tc[max_idx]
        labels_pred = crxobjs[max_idx].labels.reshape(-1)
        print 'Predicted labels are: ', labels_pred
        print 'Index for zero labels are ', np.nonzero(labels_pred == 0)
        print 'Index for one labels are ', np.nonzero(labels_pred == 1)
        if labels is not None:
            print 'True label are: ', labels
            print 'adjusted rand score: ', skm.adjusted_rand_score(labels, labels_pred)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        amr_dot_file = str(sys.argv[1])
        start_amr = int(sys.argv[2])
        end_amr = int(sys.argv[3])
        is_root_kernel = bool(sys.argv[4])
        lam = float(sys.argv[5])
        if len(sys.argv) > 6:
            cosine_threshold = float(sys.argv[6])
        else:
            cosine_threshold = None
        graph_kernels_func(amr_dot_file=amr_dot_file, start_amr=start_amr, end_amr=end_amr, lam=lam, cosine_threshold=cosine_threshold, is_root_kernel=is_root_kernel)
    else:
        cosine_threshold = float(sys.argv[1])
        test_word_kernel_pos_def(cosine_threshold)

