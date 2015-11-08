import constants_absolute_path as cap
import pickle
import save_sparse_scipy_matrices as sssm
import numpy as np
import time
import sklearn.svm as skl_svm
import config_kernel as ck
import random as r
r.seed(871227)
import eval_divergence_frm_kernel as edk
import json
import semi_automated_extraction_features_chicago_data as saefcd
import scipy.sparse.linalg as ssl
import parallel_computing as pk
import corex_topic.corex_topic as ct
import matplotlib.pyplot as plt


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
    subset_idx = r.sample(org_idx, num_subset)
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
    crx_obj_pos = ct.Corex(n_hidden=2, verbose=True)
    crx_obj_pos.fit(K_test_train.transpose()[train_pos, :])
    _, log_z_pos = crx_obj_pos.transform(K_tt, details=True)
    print 'crx_obj_pos.tcs', crx_obj_pos.tcs
    crx_obj_pos = None
    print 'log_z_pos.shape', log_z_pos.shape
    assert log_z_pos.shape[1] == 2
    assert log_z_pos.shape[0] == K_tt.shape[0]
    print 'log_z_pos.shape', log_z_pos.shape
    #
    crx_obj_neg = ct.Corex(n_hidden=2, verbose=True)
    crx_obj_neg.fit(K_test_train.transpose()[train_neg, :])
    _, log_z_neg = crx_obj_neg.transform(K_tt, details=True)
    print 'crx_obj_neg.tcs', crx_obj_neg.tcs
    crx_obj_neg = None
    print 'log_z_neg.shape', log_z_neg.shape
    assert log_z_neg.shape[1] == 2
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


def classify_wd_gaussian_process(K_train, labels_train, K_test, is_load_classifier=False):
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
    score_test_pred = K_test.dot(x[0]) + bias
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


def classify_linear_least_square(K_train, labels_train, K_test, is_load_classifier=False):
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
    print 'Learning linear least square classifications ...'
    start_time = time.time()
    print 'computing the least squares'
    #
    K_train_pos = K_train[train_pos, :]
    K_train_pos = K_train_pos.tocsc()
    K_train_pos = K_train_pos[:, train_pos]
    K_train_pos = K_train_pos.tocsr()
    #
    K_train_pos_inv = 
    #
    K_train_test_pos = K_test.transpose()[train_pos, :]
    print K_train_pos.shape
    print K_train_test_pos.shape
    K_train_test_pos = None
    K_train_pos = None
    x_pos = K_test.dot(x_pos)
    # np.savez_compressed(file_path_pos, x_pos)
    #
    K_train_neg = K_train[train_neg, :]
    K_train_neg = K_train_neg.tocsc()
    K_train_neg = K_train_neg[:, train_neg]
    K_train_neg = K_train_neg.tocsr()
    #
    K_train_test_neg = K_test.transpose()[train_neg, :]
    print K_train_neg.shape
    print K_train_test_neg.shape
    x_neg = ssl.lsqr(K_train_neg, K_train_test_neg.todense(), show=True)
    K_train_neg = None
    x_neg = K_test.dot(x_neg)
    np.savez_compressed(file_path_neg, x_neg)
    #
    print 'Time to compute the least square solution was ', time.time()-start_time
    #
    x_diff = x_pos-x_neg
    x_pos = None
    x_neg = None
    #
    # todo: entropy maximization would be more optimal though computationally expensive test set is large
    # (at sentence level, it should be fine though)
    # todo: better option would be to select a subset matrix so that entropy is positive (not required to be matrix) on subselection
    x = x_diff.diagonal()
    x_diff = None
    n = x.size
    labels = np.zeros(n)
    labels[x > 0] = 1
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
    #
    algo_options = [gp, svm, corex]
    algo = gp
    assert algo in algo_options
    #
    is_load_classifier = True
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
    amr_graphs = None
    print 'test.shape', test.shape
    #
    train = np.setdiff1d(np.arange(0, n), test)
    print 'train.shape', train.shape
    #
    if (not is_load_classifier) or is_positive_labels_only_fr_train:
        labels_train = labels[train]
        print 'labels_train.shape', labels_train.shape
    else:
        labels_train = None
    #
    if is_positive_labels_only_fr_train:
        positive_label_train_idx = np.where(labels_train == 1)
        train = train[positive_label_train_idx]
        #
        if is_load_classifier:
            labels_train = None
        else:
            labels_train = labels_train[positive_label_train_idx]
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
    print 'getting the test train matrix'
    K_test = K[test, :]
    K_test = K_test.tocsc()
    K_test = K_test[:, train]
    K_test = K_test.tocsr()
    print 'K_test.shape', K_test.shape
    print 'K_test.nnz', K_test.nnz
    #
    print 'getting the test-test matrix'
    K_tt = K[test, :]
    K_tt = K_tt.tocsc()
    K_tt = K_tt[:, test]
    K_tt = K_tt.tocsr()
    print 'K_tt.shape', K_tt.shape
    print 'K_tt.nnz', K_tt.nnz
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
        labels_test_pred_prob = classify_wd_gaussian_process(K_train, labels_train, K_test, is_load_classifier=is_load_classifier)
        labels_test_pred = np.zeros(labels_test_pred_prob.shape)
        labels_test_pred[np.where(labels_test_pred_prob > 0.5)] = 1
        labels_test_pred_prob = None
        # labels_test_pred = classify_linear_least_square(K_train, labels_train, K_test)
    elif algo == corex:
        labels_test_pred = classify_wd_corex_test_as_samples(labels_train, K_test, K_tt)
    else:
        raise AssertionError, 'No such classification algorithm.'
    #
    precision, recall = eval_inferred_labels(labels_test_pred, test, chicago_sentence_id__interactions_list_map)
    print 'precision: {}, recall: {}'.format(precision, recall)
