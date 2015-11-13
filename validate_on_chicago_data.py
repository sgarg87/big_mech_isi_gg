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
random.seed(871227)
import numpy.random as npr
import save_sparse_scipy_matrices as sssm
import math


is_opposite = False
is_binary = True
is_train_data_frm_chicago = False
if is_train_data_frm_chicago:
    fraction_chicago_train = 0.2


def get_inference_file_path():
    return 'chicago_inferred_positives.json'


def random_subset_indices(org_idx, fraction_subset_default=0.5):
    n = len(org_idx)
    num_subset = int(round(n*fraction_subset_default))
    subset_idx = random.sample(org_idx, num_subset)
    org_idx = None
    subset_idx = np.array(subset_idx)
    # subset_idx.sort()
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
        svm_clf = skl_svm.SVC(kernel='precomputed', probability=probability, verbose=ck.is_svm_verbose, class_weight='auto')
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
    num_chunks = 1
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
    labels = np.zeros(n)
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
    np.save('./log_z_pos', log_z_pos)
    np.save('./log_z_neg', log_z_neg)
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
    np.save('./log_z_pos', log_z_pos)
    np.save('./log_z_neg', log_z_neg)
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


def infer_gp_score_mcmc(K_test_train, train_weights, K_train, bias, is_multinomial=False, is_coupling=False):
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
    m = K_test_train.shape[0]
    n = K_test_train.shape[1]
    assert n == train_weights.size
    assert n == K_train.shape[0]
    assert K_train.shape[0] == K_train.shape[1]
    #
    num_mcmc_trials = 1.2*(math.log(n, 2)**3)
    #
    start_time = time.time()
    transition_matrix = np.exp(-np.asarray(K_train.todense()))
    print 'Time to compute the transition matrix ', time.time()-start_time
    #
    K_train = None
    if is_multinomial:
        if is_coupling:
            raise NotImplementedError
        else:
            random_node_sel = None
            uniform_rnd_coins = None
            uniform_rnd_coins_fr_accept = None
        #
        norm = transition_matrix.sum(1)
        norm = norm.reshape(n, 1)
        norm = np.tile(norm, n)
        transition_matrix /= norm
    else:
        start_time_coin_toss = time.time()
        random_node_sel = npr.randint(0, n-1, num_mcmc_trials)
        uniform_rnd_coins = npr.random(size=num_mcmc_trials)
        uniform_rnd_coins_fr_accept = npr.random(size=num_mcmc_trials)
        print 'time to toss the coins is ', time.time()-start_time_coin_toss
    #
    test_idx = range(m)
    test_score_pred = np.zeros(m)
    for curr_idx in test_idx:
        print '**********************************************'
        start_time = time.time()
        curr_k = np.asarray(K_test_train[curr_idx].todense()).flatten()
        print 'curr_k', curr_k.shape
        #
        curr_train_idx = sample_train_cond_test_mcmc(
            transition_matrix,
            curr_k,
            train_weights,
            is_multinomial=is_multinomial,
            is_coupling=is_coupling,
            random_node_sel=random_node_sel,
            uniform_rnd_coins=uniform_rnd_coins,
            uniform_rnd_coins_fr_accept=uniform_rnd_coins_fr_accept)
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
    test_score_pred += bias
    return test_score_pred


def sample_train_cond_test_mcmc(
        transition_matrix,
        K_train_test,
        train_weights,
        is_multinomial,
        is_coupling=False,
        random_node_sel=None,
        uniform_rnd_coins=None,
        uniform_rnd_coins_fr_accept=None):
    #
    # todo: coupling using same seed is not as efficient as coupling with prior sampling
    # todo: you may also decide on number of MCMC steps on based on convergence
    if is_coupling:
        assert random_node_sel is not None
        assert uniform_rnd_coins is not None
        assert uniform_rnd_coins_fr_accept is not None
        num_mcmc_steps = random_node_sel.size
        assert num_mcmc_steps == uniform_rnd_coins.size
        assert num_mcmc_steps == uniform_rnd_coins_fr_accept.size
    else:
        raise NotImplementedError
    #
    print transition_matrix.shape
    print K_train_test.shape
    print train_weights.shape
    #
    n = transition_matrix.shape[0]
    assert n == transition_matrix.shape[1]
    assert n == K_train_test.shape[0]
    assert n == train_weights.shape[0]
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
    curr_idx = random_node_sel[count_mcmc_steps]
    #
    while count_mcmc_steps < num_mcmc_steps-1:
        curr_node_lkl = lkl_train_nodes[curr_idx]
        #
        if is_multinomial:
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
                curr_candidate_new_idx = random_node_sel[count_mcmc_steps]
                if curr_candidate_new_idx in eval_train_idx_list:
                    continue
                if uniform_rnd_coins[count_mcmc_steps] < transition_matrix[curr_idx, curr_candidate_new_idx]:
                    new_node_idx = curr_candidate_new_idx
                    break
            assert new_node_idx is not None
        #
        new_node_lkl = lkl_train_nodes[new_node_idx]
        eval_train_idx_list.append(new_node_idx)
        #
        prob_transition \
            = min(
                1,
                (
                    (new_node_lkl*transition_matrix[new_node_idx, curr_idx])
                    /
                    float(curr_node_lkl*transition_matrix[curr_idx, new_node_idx])
                )
        )
        #
        if uniform_rnd_coins_fr_accept[count_mcmc_steps] < prob_transition:
            curr_idx = new_node_idx
            mcmc_train_idx_list.append(new_node_idx)
    #
    return eval_train_idx_list


def classify_wd_gaussian_process(K_train, labels_train, K_test, is_load_classifier=False, is_mcmc_train_cond_test=False, is_coupling=False):
    file_path = './lst_sqr_kernel_Kinv_mul_Y'
    #
    c = 3
    assert c >= 3, 'c decides probability values for train positives and negative values.' \
                   ' Any value less than 3 gives not so appropriate probabilities.'
    # bias (mean)
    # 20% positives
    bias = -0.9*c
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
        x = ssl.lsqr(K_train, (score_train-bias), show=True)
        np.savez_compressed(file_path, x)
        print 'Time to compute the least square solution was ', time.time()-start_time
        K_train = None
    else:
        x = np.load(file_path+'.npz')['arr_0']
    #
    if not is_mcmc_train_cond_test:
        score_test_pred = K_test.dot(x[0]) + bias
    else:
        assert K_train is not None
        score_test_pred = infer_gp_score_mcmc(K_test, x[0], K_train, bias, is_multinomial=False, is_coupling=is_coupling)
    #
    x = None
    K_test = None
    np.savez_compressed('./score_test_pred', score_test_pred)
    #
    print 'score_test_pred', score_test_pred
    #
    beta = 1
    labels_test_pred_prob = 1/(1+np.exp(-beta*score_test_pred))
    print 'labels_test_pred_prob', labels_test_pred_prob
    score_test_pred = None
    #
    return labels_test_pred_prob


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


def eval_inferred_labels(labels_test_pred, test, chicago_sentence_id__interactions_list_map):
    postive_sample_graphs_idx = np.where(labels_test_pred == 1)
    postive_sample_graphs_idx = test[postive_sample_graphs_idx]
    test = None
    #
    amr_graphs, _ = get_amr_data()
    amr_graphs_positive = amr_graphs[postive_sample_graphs_idx, :]
    postive_sample_graphs_idx = None
    amr_graphs = None
    #
    sentence_id_interactions_list_map = write_json_fr_interaction_str_sentence_pair_of_graphs(amr_graphs_positive, get_inference_file_path())
    #
    return evaluate_inference(sentence_id_interactions_list_map, chicago_sentence_id__interactions_list_map)


if __name__ == '__main__':
    gp = 'gp'
    svm = 'svm'
    corex = 'corex'
    lst_sqr = 'lst_sqr'
    gpr = 'gpr'
    #
    # using training data only, gives terrible performance, so use False for best results
    corex_train_data_only = False
    #
    algo_options = [gp, svm, corex, lst_sqr, gpr]
    algo = corex
    assert algo in algo_options
    #
    if algo in [gp, gpr]:
        is_mcmc_train_cond_test = True
        is_mcmc_coupling = True
    else:
        is_mcmc_train_cond_test = None
        is_mcmc_coupling = None
    #
    is_load_classifier = False
    is_positive_labels_only_fr_train = False
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
    amr_graphs = None
    print 'test.shape', test.shape
    #
    train = np.setdiff1d(np.arange(0, n), test)
    print 'train.shape', train.shape
    #
    # if (not is_load_classifier) or is_positive_labels_only_fr_train:
    labels_train = labels[train]
    print 'labels_train.shape', labels_train.shape
    # else:
    #     labels_train = None
    #
    if is_positive_labels_only_fr_train:
        positive_label_train_idx = np.where(labels_train == 1)
        train = train[positive_label_train_idx]
        #
        # if is_load_classifier:
        #     labels_train = None
        # else:
        labels_train = labels_train[positive_label_train_idx]
    #
    k_path = './graph_kernel_matrix_joint_train_data_parallel/num_cores_100.npz'
    K = sssm.load_sparse_csr(cap.absolute_path+k_path)
    print 'K.shape', K.shape
    #
    if not is_load_classifier or algo in [gp]:
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
    if algo == corex and not corex_train_data_only:
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
    if algo == svm:
        print 'SVM ...'
        _, labels_test_pred = classify_wd_svm(K_train, labels_train, K_test, is_load_classifier=is_load_classifier)
    elif algo == gp:
        print 'Gaussian process ...'
        labels_test_pred_prob \
            = classify_wd_gaussian_process(
                K_train,
                labels_train,
                K_test,
                is_load_classifier=is_load_classifier,
                is_mcmc_train_cond_test=is_mcmc_train_cond_test,
                is_coupling=is_mcmc_coupling
        )
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
            # labels_test_pred = classify_wd_corex_test_as_samples(labels_train, K_test, K_tt)
            labels_test_pred = classify_wd_corex_test_as_samples_in_chunks(labels_train, K_test, K_tt)
        else:
            labels_test_pred = classify_wd_corex_lrn_train(labels_train, K_train, K_test)
    else:
        raise AssertionError, 'No such classification algorithm.'
    #
    precision, recall = eval_inferred_labels(labels_test_pred, test, chicago_sentence_id__interactions_list_map)
    print 'precision: {}, recall: {}'.format(precision, recall)

