import constants_absolute_path as cap
import pickle
import numpy as np
import time
import sklearn.svm as skl_svm
import config_kernel as ck
import eval_divergence_frm_kernel as edk
import json
import semi_automated_extraction_features_chicago_data as saefcd
import scipy.sparse.linalg as ssl
import parallel_computing as pk
import corex_topic.corex_topic as ct
import matplotlib.pyplot as plt
import scipy.linalg as sl
import random
# random.seed(871227)
import numpy.random as npr
import save_sparse_scipy_matrices as sssm
import math
import config
import label_propagation_sparse as lps
import scipy.sparse as ss
import config_mcmc as cm
import numpy.linalg as npl


# is_opposite = False
is_binary = True
# is_train_data_frm_chicago = False
# if is_train_data_frm_chicago:
#     fraction_chicago_train = 0.2


def get_inference_file_path():
    return 'chicago_inferred_positives.json'


def random_subset_indices(org_idx, fraction_subset_default=0.5):
    n = len(org_idx)
    num_subset = int(round(n*fraction_subset_default))
    subset_idx = random.sample(org_idx, num_subset)
    org_idx = None
    subset_idx = np.array(subset_idx)
    return subset_idx


def get_chicago_test_data_idx(amr_graphs_org):
    n = amr_graphs_org.shape[0]
    assert amr_graphs_org.shape[1] == 1
    amr_graphs = []
    chicago_idx_list = []
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs_org[curr_idx, 0]
        #
        curr_path = curr_amr_graph_map['path']
        # print 'curr_path', curr_path
        #
        if 'chicago' in curr_path:
            chicago_idx_list.append(curr_idx)
    #
    chicago_idx_list = np.array(chicago_idx_list)
    return chicago_idx_list


def write_json_fr_interaction_str_sentence_pair_of_graphs(amr_graphs_org, file_path, assert_no_duplicates=False):
    n = amr_graphs_org.shape[0]
    print 'n', n
    assert amr_graphs_org.shape[1] == 1
    sentence_id_interactions_list_map = {}
    #
    count = 0
    #
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs_org[curr_idx, 0]
        #
        curr_interaction = curr_amr_graph_map['tuple']
        curr_interaction_str = edk.get_triplet_str_tuple(curr_interaction, is_concept_mapped_to_interaction=False)
        curr_interaction = None
        #
        curr_path = curr_amr_graph_map['path']
        curr_amr_graph_map = None
        curr_sentence_id = 'ID'+str(saefcd.get_passage_sentence_id_frm_joint_graph_path(curr_path))
        #
        if curr_sentence_id not in sentence_id_interactions_list_map:
            sentence_id_interactions_list_map[curr_sentence_id] = []
        if curr_interaction_str not in sentence_id_interactions_list_map[curr_sentence_id]:
            count += 1
            sentence_id_interactions_list_map[curr_sentence_id].append(curr_interaction_str)
        else:
            if assert_no_duplicates:
                raise AssertionError
    #
    print 'count', count
    with open(cap.absolute_path+file_path, 'w') as f:
        json.dump(sentence_id_interactions_list_map, f, indent=4)
    #
    return sentence_id_interactions_list_map


def get_amr_data():
    amr_pickle_file_path = './amr_data_temp.pickle'
    with open(cap.absolute_path+amr_pickle_file_path, 'rb') as f_p:
        print 'loading ...'
        amr_data = pickle.load(f_p)
        print 'loaded'
        amr_graphs = amr_data['amr']
        labels = amr_data['label']
        return amr_graphs, labels


def load_json_obj(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
        return obj


def evaluate_inference(inferred_sentence_id_interactions_list_map, chicago_sentence_id__interactions_list_map):
    if inferred_sentence_id_interactions_list_map is None:
        inferred_sentence_id_interactions_list_map = load_json_obj(get_inference_file_path())
    #
    print 'len(inferred_sentence_id_interactions_list_map)', len(inferred_sentence_id_interactions_list_map)
    num_infer = 0
    for curr_sentence_id in inferred_sentence_id_interactions_list_map:
        num_infer += len(inferred_sentence_id_interactions_list_map[curr_sentence_id])
    print 'num_infer', num_infer
    curr_sentence_id = None
    #
    print 'len(chicago_sentence_id__interactions_list_map)', len(chicago_sentence_id__interactions_list_map)
    num_chicago = 0
    for curr_sentence_id in chicago_sentence_id__interactions_list_map:
        num_chicago += len(chicago_sentence_id__interactions_list_map[curr_sentence_id])
    print 'num_chicago', num_chicago
    curr_sentence_id = None
    #
    match_count = 0
    unique_matched_chicago_interactions = []
    for curr_sentence_id in inferred_sentence_id_interactions_list_map:
        for curr_inferred_interaction in inferred_sentence_id_interactions_list_map[curr_sentence_id]:
            print 'curr_inferred_interaction', curr_inferred_interaction
            is_match = False
            if curr_sentence_id in chicago_sentence_id__interactions_list_map:
                for curr_chicago_interaction in chicago_sentence_id__interactions_list_map[curr_sentence_id]:
                    print 'curr_chicago_interaction', curr_chicago_interaction
                    if saefcd.is_match_interactions(
                            interaction_str_tuple2_chicago=curr_chicago_interaction,
                            interaction_str_tuple1_extracted=curr_inferred_interaction):
                        is_match = True
                        match_count += 1
                        curr_chicago_interaction_tuple = (curr_chicago_interaction, curr_sentence_id)
                        if curr_chicago_interaction_tuple not in unique_matched_chicago_interactions:
                            unique_matched_chicago_interactions.append(curr_chicago_interaction_tuple)
                        break
    #
    print 'match_count', match_count
    num_matched_chicago = len(unique_matched_chicago_interactions)
    print 'num_matched_chicago', num_matched_chicago
    precision = float(match_count)/num_infer
    print 'precision', precision
    recall = float(num_matched_chicago)/num_chicago
    print 'recall', recall
    #
    with open('./chicago_inferrence_results.txt', 'w') as f:
        f.write('precision:{}'.format(precision))
        f.write('\n')
        f.write('recall: {}'.format(recall))
        f.write('\n')
    #
    return precision, recall


def classify_wd_svm(K_train, labels_train, K_test, is_load_classifier=False):
    probability = False
    if not is_load_classifier:
        print 'Learning SVM classifier ...'
        start_time = time.time()
        svm_clf = skl_svm.SVC(C=0.2, kernel='precomputed', probability=probability, verbose=ck.is_svm_verbose, class_weight='auto')
        K_train = K_train.todense()
        svm_clf.fit(K_train, labels_train)
        #
        K_train = None
        labels_train = None
        print 'Learning time was ', time.time()-start_time
        print svm_clf.n_support_
        print svm_clf.support_
        print svm_clf.support_vectors_
        #
        with open(cap.absolute_path+'./svm_clf.pickle', 'wb') as f_svm_clf:
            pickle.dump(svm_clf, f_svm_clf)
    else:
        print 'loading SVM classifier ...'
        with open(cap.absolute_path+'./svm_clf.pickle', 'rb') as f_svm_clf:
            svm_clf = pickle.load(f_svm_clf)
    #
    print 'K_test.shape', K_test.shape
    m = K_test.shape[0]
    num_divisions = 100
    #
    labels_test_pred = np.zeros(m)
    if probability:
        labels_test_pred_prob = np.zeros(m)
    else:
        labels_test_pred_prob = None
    #
    test_idx_list = pk.uniform_distribute_tasks_across_cores(m, num_divisions)
    #
    for curr_division in range(num_divisions):
        curr_test_idx = test_idx_list[curr_division]
        curr_K_test = K_test[curr_test_idx, :]
        curr_K_test = curr_K_test.todense()
        labels_test_pred[curr_test_idx] = svm_clf.predict(curr_K_test)
        #
        if probability:
            labels_test_pred_prob[curr_test_idx] = svm_clf.predict_proba(curr_K_test)
    #
    K_test = None
    return labels_test_pred_prob, labels_test_pred


def classify_wd_corex(K_test):
    # Here the idea of using clustering for classification should work since percentage of positive labels is small and for many other reasons
    start_time = time.time()
    print 'Learning labels for the test samples with corex clustering itself ...'
    # assuming that K_test is a csr format matrix
    crx_obj = ct.Corex(n_hidden=2, verbose=True)
    crx_obj.fit(K_test.transpose())
    #
    print 'Learned in time: ', time.time()-start_time
    #
    path = './corex_biopathways_inference.dat'
    print 'saving the corex object at path: ', path
    crx_obj.save(path)
    print 'saved.'
    #
    labels = crx_obj.clusters
    #
    positive_labels = labels.sum()
    negative_labels = labels.size - positive_labels
    if positive_labels > negative_labels:
        labels = 1-labels
    #
    return labels


def classify_wd_corex_test_as_samples_in_chunks(labels_train, K_test_train, K_tt):
    m = K_test_train.shape[0]
    assert m == K_tt.shape[0]
    assert m == K_tt.shape[1]
    #
    num_chunks = 1000
    #
    test_labels_infer = -1*np.ones(m)
    #
    start_idx = np.arange(0, m, m/num_chunks)
    #
    for curr_idx in range(num_chunks):
        curr_start_idx = start_idx[curr_idx]
        if curr_idx == num_chunks-1:
            curr_end_idx = m
        else:
            curr_end_idx = start_idx[curr_idx+1]
        #
        curr_chunk_idx \
            = np.arange(curr_start_idx, curr_end_idx)
        #
        curr_K_test_train = K_test_train[curr_chunk_idx, :]
        #
        curr_K_tt = K_tt[curr_chunk_idx, :]
        curr_K_tt = curr_K_tt.tocsc()
        curr_K_tt = curr_K_tt[:, curr_chunk_idx]
        curr_K_tt = curr_K_tt.tocsr()
        #
        test_labels_infer[curr_chunk_idx] \
            = classify_wd_corex_test_as_samples(labels_train, curr_K_test_train, curr_K_tt)
    #
    assert np.all(test_labels_infer != -1)
    #
    return test_labels_infer


def classify_wd_corex_test_as_samples(labels_train, K_test_train, K_tt):
    n = K_tt.shape[0]
    assert n == K_tt.shape[1]
    #
    train_pos = np.where(labels_train == 1)[0]
    train_neg = np.where(labels_train == 0)[0]
    assert (train_pos.size+train_neg.size) == labels_train.size
    #
    # Here the idea of using clustering for classification should work since percentage of positive labels is small and for many other reasons
    start_time = time.time()
    print 'Learning labels for the test samples with corex clustering itself ...'
    # assuming that K_test_train is a csr format matrix
    num_var = 2
    crx_obj_pos = ct.Corex(n_hidden=num_var, verbose=True)
    crx_obj_pos.fit(K_test_train.transpose()[train_pos, :])
    _, log_z_pos = crx_obj_pos.transform(K_tt, details=True)
    print 'crx_obj_pos.tcs', crx_obj_pos.tcs
    crx_obj_pos = None
    print 'log_z_pos.shape', log_z_pos.shape
    assert log_z_pos.shape[1] == num_var
    assert log_z_pos.shape[0] == K_tt.shape[0]
    print 'log_z_pos.shape', log_z_pos.shape
    #
    crx_obj_neg = ct.Corex(n_hidden=num_var, verbose=True)
    crx_obj_neg.fit(K_test_train.transpose()[train_neg, :])
    _, log_z_neg = crx_obj_neg.transform(K_tt, details=True)
    print 'crx_obj_neg.tcs', crx_obj_neg.tcs
    crx_obj_neg = None
    print 'log_z_neg.shape', log_z_neg.shape
    assert log_z_neg.shape[1] == num_var
    assert log_z_neg.shape[0] == K_tt.shape[0]
    print 'log_z_neg.shape', log_z_neg.shape
    #
    print 'Learned in time: ', time.time()-start_time
    #
    np.save('./log_z_pos', log_z_pos)
    np.save('./log_z_neg', log_z_neg)
    #
    labels = np.zeros(n)
    #
    log_z_pos = log_z_pos.sum(1)
    print 'log_z_pos.shape', log_z_pos.shape
    print 'log_z_pos.sum()', log_z_pos.sum()
    #
    log_z_neg = log_z_neg.sum(1)
    print 'log_z_neg.shape', log_z_neg.shape
    print 'log_z_neg.sum()', log_z_neg.sum()
    #
    plt.plot(log_z_pos, log_z_neg, 'kx')
    plt.savefig('./log_z_pos_neg.pdf', dpi=300, format='pdf')
    #
    labels[np.where(log_z_neg > log_z_pos)[0]] = 1
    #
    positive_labels = labels.sum()
    print 'positive_labels', positive_labels
    negative_labels = labels.size - positive_labels
    print 'negative_labels', negative_labels
    if positive_labels > negative_labels:
        labels = 1-labels
    #
    return labels


def classify_wd_corex_lrn_train(labels_train, K_train, K_test_train):
    n = K_train.shape[0]
    assert n == K_train.shape[1]
    assert n == labels_train.size
    m = K_test_train.shape[0]
    assert n == K_test_train.shape[1]
    #
    train_pos = np.where(labels_train == 1)[0]
    train_neg = np.where(labels_train == 0)[0]
    assert (train_pos.size+train_neg.size) == labels_train.size
    #
    start_time = time.time()
    print 'Learning labels for the test samples with corex clustering itself ...'
    # assuming that K_test_train is a csr format matrix
    num_var = 2
    crx_obj_pos = ct.Corex(n_hidden=num_var, verbose=True)
    crx_obj_pos.fit(K_train[train_pos, :])
    _, log_z_pos = crx_obj_pos.transform(K_test_train, details=True)
    print 'crx_obj_pos.tcs', crx_obj_pos.tcs
    crx_obj_pos = None
    print 'log_z_pos.shape', log_z_pos.shape
    assert log_z_pos.shape[1] == num_var
    assert log_z_pos.shape[0] == K_test_train.shape[0]
    print 'log_z_pos.shape', log_z_pos.shape
    #
    crx_obj_neg = ct.Corex(n_hidden=num_var, verbose=True)
    crx_obj_neg.fit(K_train[train_neg, :])
    K_train = None
    _, log_z_neg = crx_obj_neg.transform(K_test_train, details=True)
    print 'crx_obj_neg.tcs', crx_obj_neg.tcs
    crx_obj_neg = None
    print 'log_z_neg.shape', log_z_neg.shape
    assert log_z_neg.shape[1] == num_var
    assert log_z_neg.shape[0] == K_test_train.shape[0]
    K_test_train=None
    print 'log_z_neg.shape', log_z_neg.shape
    #
    print 'Learned in time: ', time.time()-start_time
    #
    np.save('./log_z_pos', log_z_pos)
    np.save('./log_z_neg', log_z_neg)
    #
    labels = np.zeros(m)
    #
    log_z_pos = log_z_pos.sum(1)
    print 'log_z_pos.shape', log_z_pos.shape
    print 'log_z_pos.sum()', log_z_pos.sum()
    log_z_neg = log_z_neg.sum(1)
    print 'log_z_neg.shape', log_z_neg.shape
    print 'log_z_neg.sum()', log_z_neg.sum()
    #
    plt.plot(log_z_pos, log_z_neg, 'kx')
    plt.savefig('./log_z_pos_neg.pdf', dpi=300, format='pdf')
    #
    labels[np.where(log_z_neg > log_z_pos)[0]] = 1
    #
    positive_labels = labels.sum()
    print 'positive_labels', positive_labels
    negative_labels = labels.size - positive_labels
    print 'negative_labels', negative_labels
    if positive_labels > negative_labels:
        labels = 1-labels
    #
    return labels


def infer_gp_score_mcmc(
        K_test_train,
        train_weights,
        K_train,
        bias,
        is_multinomial=False,
        is_coupling=False,
        is_pure_random=False,
        is_kernel_distance=True,
        curr_seed=None
    ):
    #
    # todo: couple all of chain for test samples by sampling random selection (random nodes are selected in each MCMC step, and also random value)
    # todo: note that number so such random trials can be more than mcmc samples. So, sample more in advance and then use in sequence while MCMC sampling
    # todo: benefit of this techniques would be that all test samples would have correlated set of training samples.
    # todo: The uncorrelation would be only due to difference in their test graphs.
    # todo: So, in that sense, all test samples would have similar classifer since classifer depends on selected training points.
    # todo: such correlated classifer would be useful when doing assembly, and also when comparing regression values using positive samples and negatives samples
    # todo: also, it should help ensure that some test samples should not do too bad on train conditional test samples just because of bad random sampling.
    # todo: computational time and memory bottleneck can be reduced because of correlation in training set for sentences in a single document.
    # todo: This can help significantly for large scale learning where number of training points for the complete set are in millions. We can plan to have such a large
    # todo: data set for good evaluation of this methods. For obtaining large number of training samples, we can look on previous datasets which have only protein-protein interactions.
    # todo: For identifying interaction-type given protein-protein interactions, our system should give 80% accuracy as per the Chicago results.
    # todo: Along those lines, we can identify interaction types for all available data in public domain. And then, retrain our models.
    # todo: since training sets for different documents will turn out to be different despite the coupling.
    # todo: This information can be used to classify documents from analysis perspective.
    # todo: also, assuming that sentences from same document have natural overlap on training samples, we can run MCMC
    # todo: for shorter no. of steps and use union of all training subset from different sentences. This should reduce
    # todo: computational cost for MCMC.
    # todo: in general, it seems that primary cost in MCMC should be coin toss. So, coupling at large scale should
    # todo: save that cost.
    #
    assert curr_seed is not None, 'seed required in current experiments'
    #
    npr.seed(curr_seed)
    random.seed(curr_seed)
    #
    num_couplings = 10
    #
    m = K_test_train.shape[0]
    n = K_test_train.shape[1]
    assert n == train_weights.size
    assert n == K_train.shape[0]
    assert K_train.shape[0] == K_train.shape[1]
    #
    if not is_pure_random:
        num_mcmc_trials = 2*cm.const_num_mcmc_evaluations
    else:
        num_mcmc_trials = cm.const_num_mcmc_evaluations
    #
    start_time = time.time()
    #
    K_train_dense = np.array(K_train.todense())
    #
    if is_kernel_distance:
        transition_matrix = np.exp(-K_train_dense)
    else:
        transition_matrix = 1/(1+np.exp(-K_train_dense))
        transition_matrix[np.where(K_train_dense == 0)] = 0
    #
    K_train_dense = None
    K_train = None
    print 'Time to compute the transition matrix ', time.time()-start_time
    #
    if is_kernel_distance:
        connected_nodes_map = None
    else:
        connected_nodes_map = {}
    #
    if is_multinomial:
        if is_coupling:
            raise NotImplementedError
        else:
            norm = transition_matrix.sum(1)
            norm = norm.reshape(n, 1)
            norm = np.tile(norm, n)
            transition_matrix /= norm
    else:
        if is_coupling:
            start_time_coin_toss = time.time()
            #
            random_node_sel = npr.randint(0, n-1, (num_couplings, num_mcmc_trials))
            uniform_rnd_coins = npr.random(size=(num_couplings, num_mcmc_trials))
            uniform_rnd_coins_fr_accept = npr.random(size=(num_couplings, num_mcmc_trials))
            #
            print 'time to toss the coins is ', time.time()-start_time_coin_toss
        else:
            random_node_sel = None
            uniform_rnd_coins = None
            uniform_rnd_coins_fr_accept = None
    #
    test_idx = range(m)
    test_score_pred = np.zeros(m)
    #
    time_count_dense = 0
    #
    for curr_idx in test_idx:
        print '**********************************************'
        start_time = time.time()
        curr_k = np.asarray(K_test_train[curr_idx].todense()).flatten()
        time_count_dense += time.time()-start_time
        print 'curr_k', curr_k.shape
        #
        if not is_pure_random:
            curr_train_idx = sample_train_cond_test_mcmc(
                transition_matrix,
                curr_k,
                train_weights,
                is_multinomial=is_multinomial,
                is_coupling=is_coupling,
                random_node_sel=random_node_sel,
                uniform_rnd_coins=uniform_rnd_coins,
                uniform_rnd_coins_fr_accept=uniform_rnd_coins_fr_accept,
                connected_nodes_map=connected_nodes_map,
                is_kernel_distance=is_kernel_distance
            )
        else:
            if is_coupling:
                curr_coupling = random.randint(0, num_couplings-1)
                curr_train_idx = random_node_sel[curr_coupling]
                assert len(curr_train_idx.shape) == 1
            else:
                curr_train_idx = npr.randint(0, n-1, num_mcmc_trials)
        #
        print 'len(curr_train_idx)', len(curr_train_idx)
        print 'time to get train samples condition on test is ', time.time()-start_time
        #
        curr_train_idx = np.array(curr_train_idx)
        #
        curr_k_sel = curr_k[curr_train_idx]
        train_weights_sel = train_weights[curr_train_idx]
        curr_score_pred = curr_k_sel.dot(train_weights_sel)
        test_score_pred[curr_idx] = curr_score_pred
    #
    print 'dense time is ', time_count_dense
    test_score_pred += bias
    return test_score_pred


def infer_gp_score_bayesian_optimization(
        K_test_train,
        train_weights,
        K_train,
        K_tt,
        bias):
    # computing transition matrix prior
    # MCMC sampling using bayesian optimization
    K_train_dense = np.array(K_train.todense())
    transition_matrix = np.exp(-K_train_dense)
    K_train_dense = None
    #
    num_test = K_test_train.shape[0]
    assert num_test == K_tt.shape[0]
    assert num_test == K_tt.shape[1]
    #
    num_train = K_test_train.shape[1]
    assert num_train == train_weights.size
    assert num_train == K_train.shape[0]
    assert K_train.shape[0] == K_train.shape[1]
    #
    num_chunks = 50
    test_score_pred = np.zeros(num_test)
    start_idx = np.arange(0, num_test, num_test/num_chunks)
    for curr_idx in range(num_chunks):
        curr_start_idx = start_idx[curr_idx]
        if curr_idx == num_chunks-1:
            curr_end_idx = num_test
        else:
            curr_end_idx = start_idx[curr_idx+1]
        #
        curr_chunk_idx \
            = np.arange(curr_start_idx, curr_end_idx)
        #
        curr_K_tt = K_tt[curr_chunk_idx, :]
        curr_K_tt = curr_K_tt.tocsc()
        curr_K_tt = curr_K_tt[:, curr_chunk_idx]
        curr_K_tt = curr_K_tt.tocsr()
        #
        curr_K_test_train = K_test_train[curr_chunk_idx, :]
        #
        print 'curr_K_tt.shape', curr_K_tt.shape
        print 'curr_K_test_train.shape', curr_K_test_train.shape
        #
        test_score_pred[curr_chunk_idx] = bayesian_optimization(curr_K_tt, curr_K_test_train, train_weights, bias, transition_matrix)
    #
    return test_score_pred


def bayesian_optimization(K_tt, K_test_train, train_weights, bias, transition_matrix, is_mcmc_also=False):
    print 'K_tt.shape', K_tt.shape
    print 'K_test_train.shape', K_test_train.shape
    #
    num_test = K_tt.shape[0]
    assert num_test == K_tt.shape[1]
    assert num_test == K_test_train.shape[0]
    assert K_test_train.shape[1] == train_weights.size
    #
    test_idx = np.arange(num_test)
    test_score_pred = np.zeros(num_test)
    #
    num_train = train_weights.size
    train_weights_sparse_diag_matrix = ss.lil_matrix((num_train, num_train))
    train_weights_sparse_diag_matrix.setdiag(train_weights)
    #
    num_computations_saved = 0
    #
    train_idx = np.arange(num_train)
    #
    pi_vectors = None
    #
    num_train_total_used = 0
    #
    min_iterations = 100
    #
    for curr_idx in test_idx:
        is_surrogate_score_use = False
        curr_score_surrogate = None
        sel_train_idx = train_idx
        #
        if curr_idx > min_iterations:
            # learn linear program incrementally here
            curr_prv_test_idx = np.arange(curr_idx)
            #
            if curr_idx == min_iterations+1:
                curr_K_tt = K_tt[curr_prv_test_idx, :]
                curr_K_tt = curr_K_tt.tocsc()
                curr_K_tt = curr_K_tt[:, curr_prv_test_idx]
                curr_K_tt = curr_K_tt.tocsr()
                print curr_K_tt.shape
            else:
                curr_K_tt = next_K_tt
            #
            curr_test_score_pred_prv = test_score_pred[curr_prv_test_idx]
            print curr_test_score_pred_prv.shape
            #
            curr_K_test_prv_tests = K_tt[curr_idx, :]
            curr_K_test_prv_tests = curr_K_test_prv_tests.tocsc()
            curr_k_test_test = curr_K_test_prv_tests[:, curr_idx]
            curr_K_test_prv_tests = curr_K_test_prv_tests[:, curr_prv_test_idx]
            curr_K_test_prv_tests = curr_K_test_prv_tests.tocsr()
            print curr_K_test_prv_tests.shape
            #
            print 'stacking matrices ...'
            next_K_tt1 = ss.hstack([curr_K_tt, curr_K_test_prv_tests.transpose()])
            next_K_tt2 = ss.hstack([curr_K_test_prv_tests, curr_k_test_test])
            next_K_tt = ss.vstack([next_K_tt1, next_K_tt2])
            next_K_tt1 = None
            next_K_tt2 = None
            print 'end stacking.'
            #
            #todo: later, use Schur's lemma to use previous computations
            if curr_idx == min_iterations+1:
                start_time = time.time()
                print 'learning linear program weights in bayesian optimization paradigm ...'
                curr_weights \
                    = ssl.lsqr(
                        curr_K_tt,
                        (curr_test_score_pred_prv-bias),
                        show=True,
                        iter_lim=100,
                        damp=(1e-2*math.sqrt(curr_test_score_pred_prv.size)))[0]
                print 'time to learn was ', time.time()-start_time
                #
                # learn weights for the pi vectors here
                start_time = time.time()
                print 'learning linear program weights for selecting training points ...'
                curr_weights_pi = npl.lstsq(
                                    curr_K_tt.toarray(),
                                    pi_vectors.toarray()
                                )[0]
                print 'time to learn was ', time.time()-start_time
                print 'curr_weights_pi', curr_weights_pi
                #
            else:
                assert next_weights is not None
                assert next_weights_pi is not None
                curr_weights = next_weights
                curr_weights_pi = next_weights_pi
            #
            # start_time = time.time()
            # print 'selecting training index from linear program bayesian optimization above ...'
            # print 'curr_weights.shape', curr_weights.shape
            # print 'curr_weights', curr_weights
            # assert not np.all(curr_weights == 0)
            # curr_K_test_prv_tests_arr = curr_K_test_prv_tests.toarray()[0]
            # print 'curr_K_test_prv_tests_arr.shape', curr_K_test_prv_tests_arr.shape
            # print 'curr_K_test_prv_tests_arr', curr_K_test_prv_tests_arr
            # pi_vectors_T = pi_vectors.transpose()
            # print 'pi_vectors_T.shape', pi_vectors_T.shape
            # print 'pi_vectors_T', pi_vectors_T
            # #
            # # todo: using the weights learned from the scores for the weighted sum of pi vectors is not easy as such
            # # todo: this is supposed to give invalid values of new pi
            # curr_test__train_weights = pi_vectors_T.dot(curr_weights*curr_K_test_prv_tests_arr)
            # assert np.all(curr_test__train_weights >= 0)
            # curr_K_test_prv_tests_arr = None
            # pi_vectors_T = None
            # print 'curr_test__train_weights', curr_test__train_weights
            # curr_test__train_prob_weights = 1-np.exp(-curr_test__train_weights)
            # assert np.all(curr_test__train_prob_weights >= 0)
            # print 'curr_test__train_prob_weights', curr_test__train_prob_weights
            # curr_test__train_weights = None
            # curr_test__train_prob_weights_sum = curr_test__train_prob_weights.sum()
            # print 'curr_test__train_prob_weights_sum', curr_test__train_prob_weights_sum
            # #
            # if curr_test__train_prob_weights_sum > 0:
            #     curr_test__train_prob_weights /= curr_test__train_prob_weights_sum
            #     curr_test__train_prob_weights_sum = None
            #     mul_vec = npr.multinomial(cm.const_num_mcmc_evaluations, curr_test__train_prob_weights, size=1)[0]
            #     sel_train_idx = np.where(mul_vec != 0)[0]
            #     print 'sel_train_idx', sel_train_idx
            #     print 'time to selecting training index with bayesian optimization was ', time.time()-start_time
            # else:
            #     sel_train_idx = None
            #
            # pi vector inference
            print curr_K_test_prv_tests.shape
            print curr_weights_pi.shape
            curr_K_test_prv_tests_arr = curr_K_test_prv_tests.toarray()[0]
            print 'curr_K_test_prv_tests_arr.shape', curr_K_test_prv_tests_arr.shape
            print 'curr_K_test_prv_tests_arr', curr_K_test_prv_tests_arr
            curr_pi_vec_surrogate = curr_K_test_prv_tests_arr.dot(curr_weights_pi)
            print 'curr_pi_vec_surrogate.shape', curr_pi_vec_surrogate.shape
            # curr_pi_vec_surrogate = curr_pi_vec_surrogate.flatten()
            # print 'curr_pi_vec_surrogate.shape', curr_pi_vec_surrogate.shape
            # assert np.all(curr_pi_vec_surrogate >= 0)
            print 'curr_pi_vec_surrogate', curr_pi_vec_surrogate
            curr_test__train_prob_weights = 1-np.exp(-np.abs(curr_pi_vec_surrogate))
            assert np.all(curr_test__train_prob_weights >= 0)
            print 'curr_test__train_prob_weights', curr_test__train_prob_weights
            curr_test__train_prob_weights_sum = curr_test__train_prob_weights.sum()
            print 'curr_test__train_prob_weights_sum', curr_test__train_prob_weights_sum
            #
            if curr_test__train_prob_weights_sum > 0:
                curr_test__train_prob_weights /= curr_test__train_prob_weights_sum
                curr_test__train_prob_weights_sum = None
                mul_vec = npr.multinomial(cm.const_num_bayesian_optimization_sel, curr_test__train_prob_weights, size=1)[0]
                sel_train_idx = np.where(mul_vec != 0)[0]
                print 'sel_train_idx', sel_train_idx
                print 'time to selecting training index with bayesian optimization was ', time.time()-start_time
            else:
                # previous implementation where all training points are selected in such case
                # sel_train_idx = train_idx
                #
                # new implementation
                # since the previous test points are not correlated to the current one,
                # all training points relevant to the previous test points should hardly
                # be relevant for the current test point. So, we can discard those training points for inference on the
                # existing test point
                # assert np.all(curr_K_test_prv_tests_arr == 0)
                curr_pi_vectors_sum = np.ones(curr_K_test_prv_tests_arr.size).dot(curr_weights_pi)
                curr_pi_vectors_sum_prob_weights_min_1 = np.exp(-np.abs(curr_pi_vectors_sum))
                assert np.all(curr_pi_vectors_sum_prob_weights_min_1 >= 0)
                curr_pi_vectors_sum_prob_weights_min_1__sum = curr_pi_vectors_sum_prob_weights_min_1.sum()
                assert curr_pi_vectors_sum_prob_weights_min_1__sum > 0
                curr_pi_vectors_sum_prob_weights_min_1 /= curr_pi_vectors_sum_prob_weights_min_1__sum
                curr_pi_vectors_sum_prob_weights_min_1__sum = None
                mul_vec_fr_prv_test_samples_min_1 \
                    = npr.multinomial(cm.const_num_bayesian_optimization_sel, curr_pi_vectors_sum_prob_weights_min_1, size=1)[0]
                sel_train_idx = np.where(mul_vec_fr_prv_test_samples_min_1 != 0)[0]
                # assert np.all(curr_pi_vectors_sum_prob_weights >= 0)
                # curr_pi_vectors_sum_prob_weights__sum = curr_pi_vectors_sum_prob_weights.sum()
                # assert curr_pi_vectors_sum_prob_weights__sum > 0
                # curr_pi_vectors_sum_prob_weights /= curr_pi_vectors_sum_prob_weights__sum
                # curr_pi_vectors_sum_prob_weights__sum = None
                # mul_vec_fr_prv_test_samples = npr.multinomial((num_train - cm.const_num_bayesian_optimization_sel), curr_pi_vectors_sum_prob_weights, size=1)[0]
                # sel_train_idx_fr_prv_test_samples = np.where(mul_vec_fr_prv_test_samples != 0)[0]
                # print 'sel_train_idx_fr_prv_test_samples.size', sel_train_idx_fr_prv_test_samples.size
                # print 'sel_train_idx_fr_prv_test_samples', sel_train_idx_fr_prv_test_samples
                # sel_train_idx = np.setdiff1d(train_idx, sel_train_idx_fr_prv_test_samples)
            curr_K_test_prv_tests_arr = None
            #
            # surrogate score evaluation
            curr_score_surrogate = curr_K_test_prv_tests.dot(curr_weights)
            print 'curr_score_surrogate', curr_score_surrogate
            curr_score_surrogate += bias
            if abs(curr_score_surrogate) > 0.5:
                is_surrogate_score_use = True
                test_score_pred[curr_idx] = curr_score_surrogate
                num_computations_saved += 1
        #
        curr_k = K_test_train[curr_idx]
        print 'curr_k.shape', curr_k.shape
        print 'curr_k', curr_k
        #
        #
        print 'sel_train_idx.size', sel_train_idx.size
        print 'sel_train_idx', sel_train_idx
        if (sel_train_idx.size > cm.const_num_mcmc_evaluations) and is_mcmc_also:
            # mcmc sampling here to do further subset selection
            curr_k_bo = curr_k.toarray()[0][sel_train_idx]
            print 'curr_k_bo.shape', curr_k_bo.shape
            train_weights_bo = train_weights[sel_train_idx]
            print 'train_weights_bo.shape', train_weights_bo.shape
            transition_matrix_bo =\
                transition_matrix[np.meshgrid(sel_train_idx, sel_train_idx, indexing='ij')]
            print 'transition_matrix_bo.shape', transition_matrix_bo.shape
            #
            start_time = time.time()
            print 'mcmc sampling ...'
            sel_train_idx_bo = sample_train_cond_test_mcmc(
                transition_matrix_bo,
                curr_k_bo,
                train_weights_bo,
                is_multinomial=False,
                is_coupling=False,
                random_node_sel=None,
                uniform_rnd_coins=None,
                uniform_rnd_coins_fr_accept=None,
                connected_nodes_map=None,
                is_kernel_distance=True
            )
            print 'sel_train_idx_bo', sel_train_idx_bo
            print 'time to mcmc was', time.time()-start_time
            #
            sel_train_idx = sel_train_idx[sel_train_idx_bo]
            sel_train_idx = np.array(sel_train_idx)
        #
        #
        not_sel_train_idx = np.setdiff1d(train_idx, sel_train_idx)
        print 'not_sel_train_idx', not_sel_train_idx
        if not_sel_train_idx.size > 0:
            curr_k[0, not_sel_train_idx] = 0
        #
        #maintain pi vectors that represent contribution of training nodes to the inference
        # curr_pi_vec = abs(curr_k.dot(train_weights_sparse_diag_matrix))
        curr_pi_vec = curr_k.dot(train_weights_sparse_diag_matrix)
        print 'curr_pi_vec.shape', curr_pi_vec.shape
        if pi_vectors is None:
            pi_vectors = curr_pi_vec
        else:
            pi_vectors = ss.vstack([pi_vectors, curr_pi_vec])
        #
        # actual score prediction using all training points
        if not is_surrogate_score_use:
            num_train_total_used += sel_train_idx.size
            #
            curr_score_pred = curr_k.dot(train_weights)
            assert curr_score_pred.size == 1
            curr_score_pred = curr_score_pred[0]
            curr_score_pred += bias
            test_score_pred[curr_idx] = curr_score_pred
        #
        if curr_idx > min_iterations:
            prv_weights = np.zeros(curr_idx+1)
            prv_weights[:-1] = curr_weights
            print 'prv_weights', prv_weights
            #
            if not is_surrogate_score_use:
                curr_k_test_test = curr_k_test_test.toarray()[0][0]
                print 'curr_k_test_test', curr_k_test_test
                #
                # surrogate
                curr_score_diff = curr_score_surrogate - curr_score_pred
                curr_score_diff = curr_score_diff[0]
                print 'curr_score_diff', curr_score_diff
                prv_weights[-1] = (-curr_score_diff)/curr_k_test_test
                next_weights = prv_weights
                # train weights
                curr_pi_vec = curr_pi_vec.toarray()[0]
                curr_pi_vec_diff = curr_pi_vec_surrogate - curr_pi_vec
                print 'curr_pi_vec_diff.shape', curr_pi_vec_diff.shape
                print 'curr_pi_vec_diff', curr_pi_vec_diff
                new_pi_vec_weight = (-curr_pi_vec_diff)/curr_k_test_test
                next_weights_pi = np.vstack([curr_weights_pi, new_pi_vec_weight])
            else:
                next_weights = prv_weights
                next_weights_pi = np.vstack([curr_weights_pi, np.zeros(num_train)])
    #
    print 'num_computations_saved', num_computations_saved
    print 'avg training sampled used', num_train_total_used/float(num_test - num_computations_saved)
    return test_score_pred


def sample_train_cond_test_mcmc(
        transition_matrix,
        K_train_test,
        train_weights,
        is_multinomial,
        is_coupling=False,
        random_node_sel=None,
        uniform_rnd_coins=None,
        uniform_rnd_coins_fr_accept=None,
        connected_nodes_map=None,
        is_kernel_distance=True):
    #
    # todo: coupling using same seed is not as efficient as coupling with prior sampling
    # todo: coupling with seed won't ensure on robustness of the approach unless seed is also sampled randomly
    # todo: you may also decide on number of MCMC steps on based on convergence
    #
    # todo: try out a different version of the jumps from a node such that there is no jump to a node with zero kernel similarity
    # todo: and for nonzero similarity, we have probability of transition = 1 / (1 + exp(-k(i,j)))
    #
    # todo: try out MCMC sampling multiple times to ensure on results
    #
    # todo: *********** important idea ***********************************************************
    # todo: to reduce computational cost, based on w*K* and for a given test sample and training example, we can develop surrogate GP function to avoid many calculations.
    # todo: This can be further bounded by exploiting correlation between test samples.
    #
    # todo: using coupling idea, maximize absolute of weight*k* and minimize the predicitive variance.
    #
    if not is_kernel_distance:
        assert connected_nodes_map is not None, 'currently in use'
    else:
        assert connected_nodes_map is None
    #
    if is_coupling:
        assert random_node_sel is not None
        assert uniform_rnd_coins is not None
        assert uniform_rnd_coins_fr_accept is not None
        num_mcmc_steps = random_node_sel.shape[1]
        assert num_mcmc_steps == uniform_rnd_coins.shape[1]
        assert num_mcmc_steps == uniform_rnd_coins_fr_accept.shape[1]
        #
        num_couplings = random_node_sel.shape[0]
        assert num_couplings == uniform_rnd_coins.shape[0]
        assert num_couplings == uniform_rnd_coins_fr_accept.shape[0]
        #
        curr_coupling = random.randint(0, num_couplings-1)
        random_node_sel = random_node_sel[curr_coupling]
        uniform_rnd_coins = uniform_rnd_coins[curr_coupling]
        uniform_rnd_coins_fr_accept = uniform_rnd_coins_fr_accept[curr_coupling]
        assert len(random_node_sel.shape) == 1
        assert len(uniform_rnd_coins.shape) == 1
        assert len(uniform_rnd_coins_fr_accept.shape) == 1
    else:
        assert random_node_sel is None
        assert uniform_rnd_coins is None
        assert uniform_rnd_coins_fr_accept is None
    #
    print transition_matrix.shape
    print K_train_test.shape
    print train_weights.shape
    #
    n = transition_matrix.shape[0]
    assert n == transition_matrix.shape[1]
    assert n == K_train_test.shape[0]
    assert n == train_weights.shape[0]
    print 'n', n
    #
    start_time = time.time()
    lkl_train_nodes = K_train_test*np.absolute(train_weights)
    print 'Time to compute the likelihood is ', time.time()-start_time
    #
    train_weights = None
    K_train_test = None
    #
    #mcmc sampling
    #
    eval_train_idx_list = []
    mcmc_train_idx_list = []
    #
    count_mcmc_steps = 0
    #
    if is_coupling:
        curr_idx = random_node_sel[count_mcmc_steps]
    else:
        curr_idx = random.randint(0, n-1)
    #
    while len(eval_train_idx_list) < min(cm.const_num_mcmc_evaluations, n):
        print '*****'
        curr_node_lkl = lkl_train_nodes[curr_idx]
        print 'curr_node_lkl', curr_node_lkl
        #
        if is_multinomial:
            raise NotImplementedError, 'old code, recent changes may not be suitable for this clause'
            # sample new node as per transition probability
            curr_prob = transition_matrix[curr_idx]
            new_node_idx = None
            while True:
                curr_z = npr.multinomial(1, curr_prob)
                curr_candidate_new_idx = np.where(curr_z == 1)[0][0]
                curr_z = None
                if curr_candidate_new_idx in eval_train_idx_list:
                    continue
                else:
                    new_node_idx = curr_candidate_new_idx
                    curr_prob = None
                    break
            assert new_node_idx is not None
        else:
            # importance sampling like approximation to the multinomial
            new_node_idx = None
            while True:
                count_mcmc_steps += 1
                #
                if is_coupling and (count_mcmc_steps == num_mcmc_steps-1):
                    return eval_train_idx_list
                #
                if is_coupling:
                    curr_candidate_new_idx = random_node_sel[count_mcmc_steps]
                else:
                    if not is_kernel_distance:
                        if curr_idx not in connected_nodes_map:
                            curr_choices = np.where(transition_matrix[curr_idx] != 0)[0]
                            connected_nodes_map[curr_idx] = curr_choices
                        else:
                            curr_choices = connected_nodes_map[curr_idx]
                        #
                        if config.debug:
                            print '({}, {})'.format(curr_idx, curr_choices.size)
                        curr_candidate_new_idx = npr.choice(curr_choices, 1)[0]
                        if config.debug:
                            print 'curr_candidate_new_idx', curr_candidate_new_idx
                    else:
                        curr_candidate_new_idx = random.randint(0, n-1)
                #
                print 'curr_candidate_new_idx', curr_candidate_new_idx
                #
                if curr_candidate_new_idx in eval_train_idx_list:
                    continue
                #
                if is_coupling:
                    curr_unf_rnd_coin = uniform_rnd_coins[count_mcmc_steps]
                else:
                    curr_unf_rnd_coin = random.random()
                #
                print 'curr_unf_rnd_coin', curr_unf_rnd_coin
                #
                if config.debug:
                    print 'curr_unf_rnd_coin', curr_unf_rnd_coin
                    print 'transition_matrix[curr_idx, curr_candidate_new_idx]', transition_matrix[curr_idx, curr_candidate_new_idx]
                if curr_unf_rnd_coin < transition_matrix[curr_idx, curr_candidate_new_idx]:
                    new_node_idx = curr_candidate_new_idx
                    break
                print 'new_node_idx', new_node_idx
            assert new_node_idx is not None
        #
        new_node_lkl = lkl_train_nodes[new_node_idx]
        print 'new_node_lkl', new_node_lkl
        #
        if new_node_idx not in eval_train_idx_list:
            eval_train_idx_list.append(new_node_idx)
        #
        if config.debug:
            print 'curr_node_lkl', curr_node_lkl
            print 'new_node_lkl', new_node_lkl
        #
        curr_ratio \
            = (new_node_lkl*transition_matrix[new_node_idx, curr_idx])/float(curr_node_lkl*transition_matrix[curr_idx, new_node_idx])
        curr_ratio /= 0.1
        prob_transition = min(1, curr_ratio)
        print 'prob_transition', prob_transition
        #
        if config.debug:
            print 'prob_transition', prob_transition
        #
        if is_coupling:
            curr_uniform_rnd_coins_fr_accept = uniform_rnd_coins_fr_accept[count_mcmc_steps]
        else:
            curr_uniform_rnd_coins_fr_accept = random.random()
        #
        print 'curr_uniform_rnd_coins_fr_accept', curr_uniform_rnd_coins_fr_accept
        #
        if not is_kernel_distance:
            # high temperature mode, so 0.10 ratio. This will increase no. of accepted mcmc samples.
            if prob_transition > 0.1:
                prob_transition = 1
        #
        if curr_uniform_rnd_coins_fr_accept < prob_transition:
            curr_idx = new_node_idx
            if new_node_idx not in mcmc_train_idx_list:
                mcmc_train_idx_list.append(new_node_idx)
        print 'curr_idx', curr_idx
    #
    print 'len(mcmc_train_idx_list)', len(mcmc_train_idx_list)
    print 'eval_train_idx_list', eval_train_idx_list
    return eval_train_idx_list


def classify_wd_gaussian_process_adjust_inference_lst_sqr_correlation_test(
    K_train,
    labels_train,
    K_test,
    K_tt,
    is_load_classifier=False,
    is_mcmc_train_cond_test=False,
    is_coupling=False,
    is_pure_random=False,
    curr_seed=None
):
    scores_test = classify_wd_gaussian_process(
        K_train,
        labels_train,
        K_test,
        is_load_classifier=is_load_classifier,
        is_mcmc_train_cond_test=is_mcmc_train_cond_test,
        is_coupling=is_coupling,
        is_pure_random=is_pure_random,
        curr_seed=curr_seed,
        is_score_only=True
    )
    #
    print 'computing posterior covariance ...'
    print 'pseudo inverse'
    K_train_inv = sl.pinvh(K_train.todense(), check_finite=False)
    K_train = None
    K_train_inv = ss.csr_matrix(K_train_inv)
    print 'K_train_inv.nnz', K_train_inv.nnz
    #
    print 'first dot'
    K_expr = K_test.dot(K_train_inv)
    K_train_inv = None
    K_expr = ss.csr_matrix(K_expr)
    print 'K_expr.nnz', K_expr.nnz
    #
    print 'second dot'
    K_expr = K_expr.dot(K_test.transpose())
    K_test = None
    K_expr = ss.csr_matrix(K_expr)
    print 'K_expr.nnz', K_expr.nnz
    #
    print 'difference'
    K_tt = K_tt - K_expr
    print 'K_tt.nnz', K_tt.nnz
    #
    # todo: also try predictive covariance instead
    m = scores_test.size
    num_chunks = 10
    #
    test_weights = np.zeros(m)
    #
    start_idx = np.arange(0, m, m/num_chunks)
    #
    for curr_idx in range(num_chunks):
        curr_start_idx = start_idx[curr_idx]
        if curr_idx == num_chunks-1:
            curr_end_idx = m
        else:
            curr_end_idx = start_idx[curr_idx+1]
        #
        curr_chunk_idx \
            = np.arange(curr_start_idx, curr_end_idx)
        #
        curr_K_tt = K_tt[curr_chunk_idx, :]
        curr_K_tt = curr_K_tt.tocsc()
        curr_K_tt = curr_K_tt[:, curr_chunk_idx]
        curr_K_tt = curr_K_tt.tocsr()
        #
        curr_size = curr_chunk_idx.size
        #
        test_weights[curr_chunk_idx] \
            = ssl.lsqr(curr_K_tt, scores_test[curr_chunk_idx], show=True, damp=(1e-2*math.sqrt(curr_size)))[0]
    #
    lst_square_scores = K_tt.dot(test_weights)
    lst_sqr_labels_pos_prob = get_prob_pos_label_frm_scores(lst_square_scores)
    return lst_sqr_labels_pos_prob


def get_prob_pos_label_frm_scores(score_test_pred):
    beta = 1
    labels_test_pred_prob = 1/(1+np.exp(-beta*score_test_pred))
    print 'labels_test_pred_prob', labels_test_pred_prob
    score_test_pred = None
    #
    return labels_test_pred_prob


def classify_wd_gaussian_process(
        K_train,
        labels_train,
        K_test,
        K_tt,
        is_load_classifier=False,
        is_mcmc_train_cond_test=False,
        is_coupling=False,
        is_pure_random=False,
        curr_seed=None,
        is_score_only=False,
        bias_coeff=0.33):
    file_path = './lst_sqr_kernel_Kinv_mul_Y'
    #
    c = 3
    assert c >= 3, 'c decides probability values for train positives and negative values.' \
                   ' Any value less than 3 gives not so appropriate probabilities.'
    # bias (mean)
    # 20% positives
    bias = -bias_coeff*c
    #
    if not is_load_classifier:
        print 'Learning Gaussian process classifications ...'
        score_train = -c*np.ones(labels_train.shape)
        score_train[np.where(labels_train == 1)] = c
        print 'score_train', score_train
        labels_train = None
        #
        start_time = time.time()
        print 'computing the least squares'
        assert K_train.shape[0] == K_train.shape[1]
        #
        # n = K_train.shape[0]
        # K_noise = 0.01*ss.eye(n)
        # K_train += K_noise
        #
        # todo: this lsqr algorithm is parallelizable, so do the needful
        # see the classical paper LSQR An algrithm for sparse linear equations and sparse least squares.pdf
        x = ssl.lsqr(K_train, (score_train-bias), show=True, iter_lim=100*score_train.size, damp=1e1)
        np.savez_compressed(file_path, x)
        print 'Time to compute the least square solution was ', time.time()-start_time
        if not is_mcmc_train_cond_test:
            K_train = None
    else:
        x = np.load(file_path+'.npz')['arr_0']
    #
    if not is_mcmc_train_cond_test:
        score_test_pred = K_test.dot(x[0]) + bias
    else:
        assert K_train is not None
        assert K_tt is not None
        score_test_pred \
            = infer_gp_score_bayesian_optimization(
                K_test,
                x[0],
                K_train,
                K_tt,
                bias)
    #
    x = None
    K_test = None
    np.savez_compressed('./score_test_pred', score_test_pred)
    #
    print 'score_test_pred', score_test_pred
    #
    if not is_score_only:
        labels_test_pred_prob = get_prob_pos_label_frm_scores(score_test_pred)
        return labels_test_pred_prob
    else:
        return score_test_pred


def classify_wd_gaussian_process_pos_neg(K_train, labels_train, K_test, is_load_classifier=False, is_mcmc_train_cond_test=False, is_coupling=False):
    file_path_pos = './lst_sqr_kernel_Kinv_mul_Y_pos'
    file_path_neg = './lst_sqr_kernel_Kinv_mul_Y_neg'
    #
    c = 3
    assert c >= 3, 'c decides probability values for train positives and negative values.' \
                   ' Any value less than 3 gives not so appropriate probabilities.'
    #
    bias_pos = -0.9*c
    bias_neg = 0.9*c
    #
    train_pos = np.where(labels_train == 1)[0]
    train_neg = np.where(labels_train == 0)[0]
    assert (train_pos.size+train_neg.size) == labels_train.size
    #
    if not is_load_classifier:
        print 'Learning Gaussian process regressors ...'
        score_train = c*np.ones(labels_train.shape)
        print 'score_train', score_train
        labels_train = None
        #
        assert K_train.shape[0] == K_train.shape[1]
        #
        start_time = time.time()
        print 'computing the least squares for positive'
        # todo: this lsqr algorithm is parallelizable, so do the needful
        # see the classical paper LSQR An algrithm for sparse linear equations and sparse least squares.pdf
        curr_K = K_train[train_pos]
        curr_K = curr_K.tocsc()
        curr_K = curr_K[:, train_pos]
        curr_K = curr_K.tocsr()
        x_pos = ssl.lsqr(curr_K, (score_train[train_pos]-bias_pos), show=True)
        np.savez_compressed(file_path_pos, x_pos)
        print 'Time to compute the positive least square solution was ', time.time()-start_time
        #
        start_time = time.time()
        print 'computing the least squares for negatives'
        # todo: this lsqr algorithm is parallelizable, so do the needful
        # see the classical paper LSQR An algrithm for sparse linear equations and sparse least squares.pdf
        curr_K = K_train[train_neg]
        curr_K = curr_K.tocsc()
        curr_K = curr_K[:, train_neg]
        curr_K = curr_K.tocsr()
        x_neg = ssl.lsqr(curr_K, (score_train[train_neg]-bias_neg), show=True)
        np.savez_compressed(file_path_neg, x_neg)
        print 'Time to compute the negative least square solution was ', time.time()-start_time
        #
        K_train = None
    #
    else:
        x_pos = np.load(file_path_pos+'.npz')['arr_0']
        x_neg = np.load(file_path_neg+'.npz')['arr_0']
    #
    if not is_mcmc_train_cond_test:
        score_test_pred_pos = K_test.tocsc()[:, train_pos].dot(x_pos[0]) + bias_pos
        x_pos = None
        score_test_pred_neg = K_test.tocsc()[:, train_neg].dot(x_neg[0]) + bias_neg
        x_neg = None
    else:
        assert K_train is not None
        curr_K = K_train[train_pos]
        curr_K = curr_K.tocsc()
        curr_K = curr_K[:, train_pos]
        curr_K = curr_K.tocsr()
        score_test_pred_pos \
            = infer_gp_score_mcmc(K_test.tocsc()[:, train_pos].tocsr(), x_pos[0], curr_K, bias=bias_pos, is_multinomial=False, is_coupling=is_coupling)
        x_pos = None
        #
        curr_K = K_train[train_neg]
        curr_K = curr_K.tocsc()
        curr_K = curr_K[:, train_neg]
        curr_K = curr_K.tocsr()
        score_test_pred_neg \
            = infer_gp_score_mcmc(K_test.tocsc()[:, train_neg].tocsr(), x_neg[0], curr_K, bias=bias_neg, is_multinomial=False, is_coupling=is_coupling)
        x_neg = None
    #
    K_test = None
    np.savez_compressed('./score_test_pred_pos', score_test_pred_pos)
    np.savez_compressed('./score_test_pred_neg', score_test_pred_neg)
    #
    print 'score_test_pred_pos', score_test_pred_pos
    print 'score_test_pred_neg', score_test_pred_neg
    #
    assert score_test_pred_pos.size == score_test_pred_neg.size
    labels_test_pred = np.zeros(score_test_pred_pos.size)
    labels_test_pred[score_test_pred_neg < 2.4] = 1
    labels_test_pred[score_test_pred_pos > 0] = 1
    print 'labels_test_pred', labels_test_pred
    score_test_pred_pos = None
    score_test_pred_neg = None
    #
    return labels_test_pred


def classify_linear_least_square(K_train, labels_train, K_test, is_load_classifier=False):
    if is_load_classifier:
        assert K_train is None
    else:
        assert K_train is not None
    #
    if not is_load_classifier:
        assert K_train.shape[0] == K_train.shape[1]
    #
    train_pos = np.where(labels_train == 1)[0]
    train_neg = np.where(labels_train == 0)[0]
    assert (train_pos.size+train_neg.size) == labels_train.size
    labels_train = None
    #
    file_path_pos = './lst_sqr_pos_Kinv'
    file_path_neg = './lst_sqr_neg_Kinv'
    #
    K_train_test_pos = K_test.transpose()[train_pos, :]
    print K_train_test_pos.shape
    #
    if not is_load_classifier:
        K_train_pos = K_train[train_pos, :]
        K_train_pos = K_train_pos.tocsc()
        K_train_pos = K_train_pos[:, train_pos]
        K_train_pos = K_train_pos.tocsr()
        print K_train_pos.shape
        #
        start_time = time.time()
        print 'Learning pseudo inverse of K_train_pos ...'
        K_train_pos_inv = sl.pinvh(K_train_pos.todense(), check_finite=False)
        print 'Time to learn the inverse was ', time.time()-start_time
        #
        K_train_pos = None
        np.savez_compressed(file_path_pos, K_train_pos_inv)
    else:
        K_train_pos_inv = np.load(file_path_pos+'.npz')['arr_0']
    #
    start_time = time.time()
    print 'computing dot product for positives ...'
    x_pos = K_train_test_pos.transpose().dot(K_train_pos_inv)
    K_train_pos_inv = None
    x_pos = x_pos.dot(K_train_test_pos)
    K_train_test_pos = None
    x_pos_sum = x_pos.sum(1)
    x_pos_diag = x_pos.diagonal()
    x_pos = None
    print 'computation time was ', time.time()-start_time
    #
    # training using negative examples
    K_train_test_neg = K_test.transpose()[train_neg, :]
    print K_train_test_neg.shape
    #
    if not is_load_classifier:
        K_train_neg = K_train[train_neg, :]
        K_train_neg = K_train_neg.tocsc()
        K_train_neg = K_train_neg[:, train_neg]
        K_train_neg = K_train_neg.tocsr()
        print K_train_neg.shape
        #
        start_time = time.time()
        print 'Learning pseudo inverse of K_train_neg ...'
        K_train_neg_inv = sl.pinvh(K_train_neg.todense(), check_finite=False)
        print 'Time to learn the inverse was ', time.time()-start_time
        #
        K_train_neg = None
        np.savez_compressed(file_path_neg, K_train_neg_inv)
    else:
        K_train_neg_inv = np.load(file_path_neg+'.npz')['arr_0']
    #
    start_time = time.time()
    print 'computing dot product for negatives ...'
    x_neg = K_train_test_neg.transpose().dot(K_train_neg_inv)
    K_train_neg_inv = None
    x_neg = x_neg.dot(K_train_test_neg)
    K_train_test_neg = None
    x_neg_diag = x_neg.diagonal()
    x_neg_sum = x_neg.sum(1)
    x_neg = None
    print 'computation time was ', time.time()-start_time
    #
    x_sum = x_neg_sum-x_pos_sum
    x_neg_sum = None
    x_pos_sum = None
    np.save('./x_sum', x_sum)
    #
    x_diag = x_neg_diag - x_pos_diag
    x_neg_diag = None
    x_pos_diag = None
    np.save('./x_diag', x_diag)
    #
    # todo: entropy maximization would be more optimal though computationally expensive test set is large
    # (at sentence level, it should be fine though)
    # todo: better option would be to select a subset matrix so that entropy is positive (not required to be matrix) on subselection
    #
    n = x_sum.size
    assert n == x_diag.size
    labels = np.zeros(n)
    labels[np.abs(2*x_diag) >= np.abs(x_sum)] = 1
    x_sum = None
    labels[x_diag < 0] = 0
    #
    positive_labels = labels.sum()
    print 'positive_labels', positive_labels
    negative_labels = labels.size - positive_labels
    print 'negative_labels', negative_labels
    if positive_labels > negative_labels:
        labels = 1-labels
    return labels


def classify_lsqr_sparse(K_train, labels_train, K_test, is_load_classifier=False):
    if is_load_classifier:
        assert K_train is None
    else:
        assert K_train is not None
    #
    if not is_load_classifier:
        assert K_train.shape[0] == K_train.shape[1]
    #
    train_pos = np.where(labels_train == 1)[0]
    train_neg = np.where(labels_train == 0)[0]
    assert (train_pos.size+train_neg.size) == labels_train.size
    labels_train = None
    #
    file_path_pos = './lsqr_sparse_pos'
    file_path_neg = './lsqr_sparse_neg'
    #
    K_train_test_pos = K_test.transpose()[train_pos, :]
    print K_train_test_pos.shape
    #
    if not is_load_classifier:
        K_train_pos = K_train[train_pos, :]
        K_train_pos = K_train_pos.tocsc()
        K_train_pos = K_train_pos[:, train_pos]
        K_train_pos = K_train_pos.tocsr()
        print K_train_pos.shape
        #
        start_time = time.time()
        print 'Learning sparse linear program for positives ...'
        K_train_pos_inv_mul_K_train_test = ssl.spsolve(K_train_pos.tocsc(), K_train_test_pos.tocsc())
        K_train_pos = None
        if K_train_pos_inv_mul_K_train_test.getformat() == 'csr':
            # do nothing
            pass
        elif K_train_pos_inv_mul_K_train_test.getformat() == 'csc':
            K_train_pos_inv_mul_K_train_test = K_train_pos_inv_mul_K_train_test.tocsr()
        else:
            raise AssertionError, K_train_pos_inv_mul_K_train_test.getformat()
        #
        print 'Time to learn was ', time.time()-start_time
        sssm.save_sparse_csr(file_path_pos, K_train_pos_inv_mul_K_train_test)
    else:
        K_train_pos_inv_mul_K_train_test = sssm.load_sparse_csr(file_path_pos+'.npz')
    #
    start_time = time.time()
    print 'computing dot product for positives ...'
    x_pos = K_train_test_pos.transpose().dot(K_train_pos_inv_mul_K_train_test)
    K_train_pos_inv_mul_K_train_test = None
    K_train_test_pos = None
    x_pos_sum = x_pos.sum(1)
    x_pos_diag = x_pos.diagonal()
    x_pos = None
    print 'computation time was ', time.time()-start_time
    #
    # training using negative examples
    K_train_test_neg = K_test.transpose()[train_neg, :]
    print K_train_test_neg.shape
    #
    if not is_load_classifier:
        K_train_neg = K_train[train_neg, :]
        K_train_neg = K_train_neg.tocsc()
        K_train_neg = K_train_neg[:, train_neg]
        K_train_neg = K_train_neg.tocsr()
        print K_train_neg.shape
        #
        start_time = time.time()
        print 'Learning pseudo inverse of K_train_neg ...'
        K_train_neg_inv_mul_K_train_test = ssl.spsolve(K_train_neg.tocsc(), K_train_test_neg.tocsc())
        K_train_neg = None
        if K_train_neg_inv_mul_K_train_test.getformat() == 'csr':
            print 'do nothing'
            pass
        elif K_train_neg_inv_mul_K_train_test.getformat() == 'csc':
            K_train_neg_inv_mul_K_train_test = K_train_neg_inv_mul_K_train_test.tocsr()
        else:
            raise AssertionError, K_train_neg_inv_mul_K_train_test.getformat()
        #
        print 'Time to learn the inverse was ', time.time()-start_time
        #
        sssm.save_sparse_csr(file_path_neg, K_train_neg_inv_mul_K_train_test)
    else:
        K_train_neg_inv_mul_K_train_test = sssm.load_sparse_csr(file_path_neg+'.npz')
    #
    start_time = time.time()
    print 'computing dot product for negatives ...'
    x_neg = K_train_test_neg.transpose().dot(K_train_neg_inv_mul_K_train_test)
    K_train_neg_inv_mul_K_train_test = None
    K_train_test_neg = None
    x_neg_sum = x_neg.sum(1)
    x_neg_diag = x_neg.diagonal()
    x_neg = None
    print 'computation time was ', time.time()-start_time
    #
    x_sum = x_neg_sum-x_pos_sum
    x_neg_sum = None
    x_pos_sum = None
    np.save('./x_sum', x_sum)
    #
    x_diag = x_neg_diag - x_pos_diag
    x_neg_diag = None
    x_pos_diag = None
    np.save('./x_diag', x_diag)
    #
    # todo: entropy maximization would be more optimal though computationally expensive test set is large
    # (at sentence level, it should be fine though)
    # todo: better option would be to select a subset matrix so that entropy is positive (not required to be matrix) on subselection
    #
    n = x_sum.size
    assert n == x_diag.size
    labels = np.zeros(n)
    labels[np.abs(2*x_diag) >= np.abs(x_sum)] = 1
    x_sum = None
    labels[x_diag < 0] = 0
    #
    positive_labels = labels.sum()
    print 'positive_labels', positive_labels
    negative_labels = labels.size - positive_labels
    print 'negative_labels', negative_labels
    if positive_labels > negative_labels:
        labels = 1-labels
    return labels


def eval_inferred_labels(labels_test_pred, test, chicago_sentence_id__interactions_list_map, amr_graphs=None):
    postive_sample_graphs_idx = np.where(labels_test_pred == 1)
    postive_sample_graphs_idx = test[postive_sample_graphs_idx]
    test = None
    #
    if amr_graphs is None:
        amr_graphs, _ = get_amr_data()
    amr_graphs_positive = amr_graphs[postive_sample_graphs_idx, :]
    postive_sample_graphs_idx = None
    amr_graphs = None
    #
    sentence_id_interactions_list_map = write_json_fr_interaction_str_sentence_pair_of_graphs(amr_graphs_positive, get_inference_file_path())
    #
    return evaluate_inference(sentence_id_interactions_list_map, chicago_sentence_id__interactions_list_map)


def classify_wd_label_propagation(K_train_train, K_test_train, Ktt, labels_train):
    print 'classifying with label propagation ...'
    #
    m = K_train_train.shape[0]
    assert m == K_train_train.shape[1]
    n = Ktt.shape[0]
    assert n == Ktt.shape[1]
    assert n == K_test_train.shape[0]
    assert m == K_test_train.shape[1]
    assert m == labels_train.size
    d_type = K_train_train.dtype
    assert d_type == K_test_train.dtype
    assert d_type == Ktt.dtype
    #
    p = m+n
    #
    K1 = ss.hstack([K_train_train, K_test_train.transpose()])
    K_train_train = None
    K2 = ss.hstack([K_test_train, Ktt])
    Ktt = None
    K_test_train = None
    K = ss.vstack([K1, K2])
    K1 = None
    K2 = None
    #
    print K
    sssm.save_sparse_csr('./K', K)
    #
    train = np.arange(m)
    test = np.arange(m, m+n)
    labels_test_pred_prob = lps.infer_labels_fr_test(K, labels_train, train, test)
    n = labels_test_pred_prob.shape[0]
    assert labels_test_pred_prob.shape[1] == 2
    return labels_test_pred_prob[:, 1]


def get_model_data_idx(amr_graphs_org):
    n = amr_graphs_org.shape[0]
    assert amr_graphs_org.shape[1] == 1
    model_idx_list = []
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs_org[curr_idx, 0]
        #
        curr_path = curr_amr_graph_map['path']
        #
        if 'json_darpa_model_amrs' in curr_path:
            model_idx_list.append(curr_idx)
    #
    model_idx_list = np.array(model_idx_list)
    return model_idx_list


def get_canonical_unique_model(amr_graphs_org):
    n = amr_graphs_org.shape[0]
    assert amr_graphs_org.shape[1] == 1
    canonical_model_idx_list = []
    canonical_tuples_str_list = []
    #
    for curr_amr_idx in range(n):
        # print '*************************************'
        # print 'curr_amr_idx', curr_amr_idx
        curr_amr_graph_map = amr_graphs_org[curr_amr_idx, 0]
        #
        curr_path = curr_amr_graph_map['path']
        curr_tuple = curr_amr_graph_map['tuple']
        curr_tuple_str = edk.get_triplet_str_tuple(curr_tuple)
        curr_tuple = None
        curr_tuple_str = list(curr_tuple_str)
        #
        # canonical transformation
        for curr_idx in range(1, len(curr_tuple_str)):
            if curr_tuple_str[curr_idx] is not None:
                curr_tuple_str[curr_idx] = 'A'
        curr_idx = None
        #
        curr_tuple_str = tuple(curr_tuple_str)
        # print 'curr_tuple_str', curr_tuple_str
        #
        if curr_tuple_str not in canonical_tuples_str_list:
            # print 'adding this one ..'
            canonical_tuples_str_list.append(curr_tuple_str)
            canonical_model_idx_list.append(curr_amr_idx)
    #
    canonical_model_idx_list = np.array(canonical_model_idx_list)
    print 'canonical_model_idx_list', canonical_model_idx_list
    print 'canonical_tuples_str_list', canonical_tuples_str_list
    return canonical_model_idx_list


if __name__ == '__main__':
    gp = 'gp'
    svm = 'svm'
    corex = 'corex'
    lst_sqr = 'lst_sqr'
    gpr = 'gpr'
    lp = 'lp'
    #
    # using training data only
    corex_train_data_only = True
    #
    algo_options = [gp, svm, corex, lst_sqr, gpr, lp]
    algo = gp
    assert algo in algo_options
    #
    if algo in [gp, gpr]:
        is_mcmc_train_cond_test = True
        is_mcmc_coupling = False
    else:
        is_mcmc_train_cond_test = None
        is_mcmc_coupling = None
    #
    if algo == gp:
        is_test_correlation_in_gp = False
    else:
        is_test_correlation_in_gp = None
    #
    is_load_classifier = False
    is_positive_labels_only_fr_train = True
    #
    amr_graphs, labels = get_amr_data()
    #
    n = labels.size
    print 'labels.size', labels.size
    #
    if is_binary:
        idx_label2 = np.where(labels == 2)
        labels[idx_label2] = 0
    #
    test = get_chicago_test_data_idx(amr_graphs)
    # amr_graphs = None
    print 'test.shape', test.shape
    #
    train = np.setdiff1d(np.arange(0, n), test)
    amr_graphs_train = amr_graphs[train, :]
    print 'amr_graphs_train.shape', amr_graphs_train.shape
    train_model = get_model_data_idx(amr_graphs_train)
    amr_graphs_train = None
    train_model = train[train_model]
    #
    amr_graphs_train_model = amr_graphs[train_model, :]
    train_model_canonical = get_canonical_unique_model(amr_graphs_train_model)
    amr_graphs_train_model = None
    train_model_canonical = train_model[train_model_canonical]
    train_model = np.setdiff1d(train_model, train_model_canonical)
    #
    train_not_model = np.setdiff1d(train, train_model)
    train_model = None
    train = train_not_model
    print 'train.shape', train.shape
    #
    # if (not is_load_classifier) or is_positive_labels_only_fr_train:
    labels_train = labels[train]
    print 'labels_train.shape', labels_train.shape
    # else:
    #     labels_train = None
    #
    if is_positive_labels_only_fr_train:
        print 'positives only ...'
        positive_label_train_idx = np.where(labels_train == 1)
        train = train[positive_label_train_idx]
        print 'train.shape', train.shape
        #
        # if is_load_classifier:
        #     labels_train = None
        # else:
        labels_train = labels_train[positive_label_train_idx]
        print 'labels_train.shape', labels_train.shape
    #
    k_path = './graph_kernel_matrix_joint_train_data_parallel/num_cores_100.npz'
    K = sssm.load_sparse_csr(cap.absolute_path+k_path)
    print 'K.shape', K.shape
    #
    if not is_load_classifier or algo in [gp, lp]:
        K_train = K[train, :]
        K_train = K_train.tocsc()
        K_train = K_train[:, train]
        K_train = K_train.tocsr()
        print 'K_train.shape', K_train.shape
        print 'K_train.nnz', K_train.nnz
    else:
        K_train = None
    #
    print 'getting the test train matrix'
    K_test = K[test, :]
    K_test = K_test.tocsc()
    K_test = K_test[:, train]
    K_test = K_test.tocsr()
    print 'K_test.shape', K_test.shape
    print 'K_test.nnz', K_test.nnz
    #
    if (algo == corex and not corex_train_data_only) or (algo == lp) or (algo == gp):
        print 'getting the test-test matrix'
        K_tt = K[test, :]
        K_tt = K_tt.tocsc()
        K_tt = K_tt[:, test]
        K_tt = K_tt.tocsr()
        print 'K_tt.shape', K_tt.shape
        print 'K_tt.nnz', K_tt.nnz
    else:
        K_tt = None
    #
    train = None
    K = None
    #
    chicago_sentence_id__interactions_list_map \
        = load_json_obj(saefcd.get_file_path_matched_org_positive_sentence_id_interaction_list_map())
    #
    num_trials = 1
    #
    file_path = './precision_recall_trials.json'
    # f = open(file_path, 'w')
    # f.write('num_trials: {}'.format(num_trials))
    # f.write('\n\n')
    # f.close()
    f1_score_list = []
    precision_recall_list = []
    seed_list = []
    json_obj = {'seed': seed_list, 'pr': precision_recall_list, 'f1': f1_score_list}
    #
    for curr_trial in range(num_trials):
        # f = open(file_path, 'a')
        curr_seed = random.getrandbits(32)
        seed_list.append(curr_seed)
        # f.write('seed: {}'.format(curr_seed))
        # f.write('\n')
        # f.close()
        #
        if algo == svm:
            print 'SVM ...'
            _, labels_test_pred = classify_wd_svm(K_train, labels_train, K_test, is_load_classifier=is_load_classifier)
        elif algo == gp:
            print 'Gaussian process ...'
            if is_test_correlation_in_gp:
                labels_test_pred_prob \
                    = classify_wd_gaussian_process_adjust_inference_lst_sqr_correlation_test(
                        K_train,
                        labels_train,
                        K_test,
                        K_tt,
                        is_load_classifier=is_load_classifier,
                        is_mcmc_train_cond_test=is_mcmc_train_cond_test,
                        is_coupling=is_mcmc_coupling,
                        is_pure_random=False,
                        curr_seed=curr_seed
                )
            else:
                labels_test_pred_prob \
                    = classify_wd_gaussian_process(
                        K_train,
                        labels_train,
                        K_test,
                        K_tt,
                        is_load_classifier=is_load_classifier,
                        is_mcmc_train_cond_test=is_mcmc_train_cond_test,
                        is_coupling=is_mcmc_coupling,
                        is_pure_random=False,
                        curr_seed=curr_seed
                )
            #
            labels_test_pred = np.zeros(labels_test_pred_prob.shape)
            labels_test_pred[np.where(labels_test_pred_prob > 0.5)] = 1
            labels_test_pred_prob = None
        elif algo == gpr:
            print 'Gaussian process regression using positive vs negatives ...'
            labels_test_pred \
                = classify_wd_gaussian_process_pos_neg(
                    K_train,
                    labels_train,
                    K_test,
                    is_load_classifier=is_load_classifier,
                    is_mcmc_train_cond_test=is_mcmc_train_cond_test,
                    is_coupling=is_mcmc_coupling
            )
        elif algo == lst_sqr:
            print 'Least square ...'
            labels_test_pred = classify_linear_least_square(K_train, labels_train, K_test, is_load_classifier=is_load_classifier)
            # labels_test_pred = classify_lsqr_sparse(K_train, labels_train, K_test, is_load_classifier=is_load_classifier)
        elif algo == corex:
            if not corex_train_data_only:
                labels_test_pred = classify_wd_corex_test_as_samples(labels_train, K_test, K_tt)
                # labels_test_pred = classify_wd_corex_test_as_samples_in_chunks(labels_train, K_test, K_tt)
            else:
                # labels_test_pred = classify_wd_corex(K_test)
                labels_test_pred = classify_wd_corex_lrn_train(labels_train, K_train, K_test)
        elif algo == lp:
            labels_test_pred_prob = classify_wd_label_propagation(K_train, K_test, K_tt, labels_train)
            labels_test_pred = np.zeros(labels_test_pred_prob.shape)
            labels_test_pred[np.where(labels_test_pred_prob > 0.5)] = 1
            labels_test_pred_prob = None
        else:
            raise AssertionError,'No such classification algorithm.'
        #
        precision, recall \
            = eval_inferred_labels(labels_test_pred, test, chicago_sentence_id__interactions_list_map, amr_graphs=amr_graphs)
        print 'precision: {}, recall: {}'.format(precision, recall)
        f1 = (2*precision*recall)/(precision+recall)
        precision_recall_list.append({'precision': precision, 'recall': recall, 'f1': f1})
        f1_score_list.append(f1)
    #
    f1_score_list = np.array(f1_score_list)
    json_obj['mean_f1'] = f1_score_list.mean()
    json_obj['std_f1'] = f1_score_list.std()
    #
    with open(file_path, 'w') as f:
        json.dump(json_obj, f, indent=4, sort_keys=True)


