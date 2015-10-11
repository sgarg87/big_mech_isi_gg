import config_hpcc as ch
if not ch.is_hpcc:
    import matplotlib.pyplot as plt
#
import csv
import numpy as np
import math
import time
from config import *
import scipy.special as ss
import sys
import config_kernel as ck
import sklearn.svm as skl_svm
import sklearn.metrics as skm
import numpy.random as rnd
import json
import pickle
import label_propagation as lp
import amr_sets
import amr_sets_aimed
import eval_divergence_frm_kernel as edk
import constants_absolute_path as cap
import random as r
r.seed(871227)
import itertools as it
import math as m


# support vector machines
classifier = 'svm'
# label propagation
# classifier = 'lp'


def get_triplet_str_tuple(curr_triplet_nodes_tuple):
    curr_triplet_str_tuple = []
    m = len(curr_triplet_nodes_tuple)
    for curr_idx in range(m):
        if curr_triplet_nodes_tuple[curr_idx] is not None:
            curr_str = curr_triplet_nodes_tuple[curr_idx].get_name_formatted()
            curr_triplet_str_tuple.append(curr_str)
        else:
            curr_triplet_str_tuple.append(None)
    return tuple(curr_triplet_str_tuple)


def get_metrics(labels_test, labels_test_pred, labels_test_pred_prob_positive_label):
    # adjusted rand index on all labels
    adj_rand_score_test = skm.adjusted_rand_score(labels_test.astype(np.int), labels_test_pred.astype(np.int))
    # confusion matrix on all labels
    confusion_matrix_test = skm.confusion_matrix(labels_test.astype(np.int), labels_test_pred.astype(np.int))
    #
    print 'confusion_matrix_test', confusion_matrix_test
    #
    if confusion_matrix_test.size == 1:
        positive_label_precision_test = 1
        positive_label_recall_test = 1
        positive_label_f1_score_test = 1
    elif confusion_matrix_test.shape[0] == 1 and confusion_matrix_test.shape[1] > 1:
        raise NotImplementedError
    else:
        # precision evaluation on positive labels
        if confusion_matrix_test[:,1].sum() == 0:
            assert confusion_matrix_test[1, 1] == 0
            positive_label_precision_test = 1
        else:
            positive_label_precision_test = confusion_matrix_test[1, 1]/float(confusion_matrix_test[:,1].sum())
        # recall evaluation on positive labels
        if confusion_matrix_test[1, :].sum() == 0:
            assert confusion_matrix_test[1, 1] == 0
            positive_label_recall_test = 1
        else:
            positive_label_recall_test = confusion_matrix_test[1, 1]/float(confusion_matrix_test[1, :].sum())
        # f1 score evaluation on positive labels
        if (positive_label_precision_test == 0) or (positive_label_recall_test == 0):
            positive_label_f1_score_test = 0
        else:
            positive_label_f1_score_test \
                = 2*positive_label_precision_test*positive_label_recall_test/\
                  float(positive_label_precision_test + positive_label_recall_test)
    #
    labels_test_binary = np.copy(labels_test)
    labels_test_binary[np.where(labels_test_binary == 2)] = 0
    labels_test_pred_prob_binary = np.copy(labels_test_pred_prob_positive_label)
    precision_recall_curve_map = {}
    precision_recall_curve_map['precision'], precision_recall_curve_map['recall'], precision_recall_curve_map['thresholds']\
        = skm.precision_recall_curve(labels_test_binary, labels_test_pred_prob_binary, 1)
    return \
        adj_rand_score_test,\
        confusion_matrix_test,\
        positive_label_precision_test,\
        positive_label_recall_test,\
        positive_label_f1_score_test,\
        precision_recall_curve_map


def get_max_lkl_inferred_nd_true_labels_fr_interaction(test_idx, labels_inferred, labels_inferred_prob_fr_positive
                                                       , interaction_idx_list_map):
    raise DeprecationWarning
    n = test_idx.size
    print 'test_idx', test_idx
    true_labels_list = []
    inferred_labels_list_fr_max_lkl = []
    inferred_labels_max_lkl_list = []
    #
    print 'n', n
    print 'len(interaction_idx_list_map)', len(interaction_idx_list_map)
    #
    for curr_interaction_tuple in interaction_idx_list_map:
        print '******************************************'
        print 'curr_interaction_tuple', curr_interaction_tuple
        curr_true_label = curr_interaction_tuple[1]
        print 'curr_true_label', curr_true_label
        curr_interaction_idx_list = np.array(interaction_idx_list_map[curr_interaction_tuple])
        print 'curr_interaction_idx_list', curr_interaction_idx_list
        # assuming that indices in test are sorted
        assert all(test_idx[i] <= test_idx[i+1] for i in range(len(test_idx)-1))
        #
        curr_interaction_idx_list = np.intersect1d(test_idx, curr_interaction_idx_list)
        print 'curr_interaction_idx_list', curr_interaction_idx_list
        #
        curr_interaction_idx_list = np.searchsorted(test_idx, curr_interaction_idx_list)
        assert np.all(curr_interaction_idx_list < n)
        print 'curr_interaction_idx_list', curr_interaction_idx_list
        #
        if curr_interaction_idx_list.size == 0:
            continue
        #
        curr_labels_inferred = labels_inferred[curr_interaction_idx_list]
        print 'curr_labels_inferred', curr_labels_inferred
        curr_labels_inferred_prob_fr_positive = labels_inferred_prob_fr_positive[curr_interaction_idx_list]
        print 'curr_labels_inferred_prob_fr_positive', curr_labels_inferred_prob_fr_positive
        curr_label_inferred_max_lkl = curr_labels_inferred[curr_labels_inferred_prob_fr_positive.argmax()]
        print 'curr_label_inferred_max_lkl', curr_label_inferred_max_lkl
        #
        true_labels_list.append(curr_true_label)
        inferred_labels_list_fr_max_lkl.append(curr_label_inferred_max_lkl)
        inferred_labels_max_lkl_list.append(curr_labels_inferred_prob_fr_positive.max())
    #
    true_labels_list = np.array(true_labels_list)
    print 'len(true_labels_list)', len(true_labels_list)
    inferred_labels_list_fr_max_lkl = np.array(inferred_labels_list_fr_max_lkl)
    print 'inferred_labels_list_fr_max_lkl', inferred_labels_list_fr_max_lkl
    inferred_labels_max_lkl_list = np.array(inferred_labels_max_lkl_list)
    print 'inferred_labels_max_lkl_list', inferred_labels_max_lkl_list
    return true_labels_list, inferred_labels_list_fr_max_lkl, inferred_labels_max_lkl_list


def print_nd_plot_metric_summary(algo_name_fr_plots, adj_rand_score_test, confusion_matrix_test,
        positive_label_precision_score_test, positive_label_recall_score_test, positive_label_f1_score_test,
        positive_label_precision_recall_curve_map, train_test_amr_mmd_divergence, is_max_lkl_label=False):
    file_path = './validation_plots_nd_text_output/precision_recall_{}'.format(algo_name_fr_plots)
    if is_max_lkl_label:
        file_path += '_max_lkl_label'
    results_obj = {}
    #
    adjusted_rand_score_obj = {}
    results_obj['adjusted rand score'] = adjusted_rand_score_obj
    adjusted_rand_score_obj['max'] = adj_rand_score_test.max()
    adjusted_rand_score_obj['min'] = adj_rand_score_test.min()
    adjusted_rand_score_obj['mean'] = adj_rand_score_test.mean()
    adjusted_rand_score_obj['z_list'] = adj_rand_score_test.tolist()
    #
    f1_score_obj = {}
    results_obj['f1 score'] = f1_score_obj
    f1_score_obj['max'] = positive_label_f1_score_test.max()
    f1_score_obj['min'] = positive_label_f1_score_test.min()
    f1_score_obj['mean'] = positive_label_f1_score_test.mean()
    f1_score_obj['std'] = positive_label_f1_score_test.std()
    f1_score_obj['z_list'] = positive_label_f1_score_test.tolist()
    #
    precision_obj = {}
    results_obj['precision'] = precision_obj
    precision_obj['z_list'] = positive_label_precision_score_test.tolist()
    #
    recall_obj = {}
    results_obj['recall'] = recall_obj
    recall_obj['z_list'] = positive_label_recall_score_test.tolist()
    #
    confusion_matrix_obj = {}
    positive_label_percent_obj = {}
    positive_label_abs_obj = {}
    results_obj['confusion_matrix'] = confusion_matrix_obj
    results_obj['positive_label_percent'] = positive_label_percent_obj
    results_obj['positive_label_abs'] = positive_label_abs_obj
    #
    confusion_matrix_obj['z_list'] = []
    #
    positive_label_percent = -1*np.ones(shape=confusion_matrix_test.shape)
    positive_label_abs = -1*np.ones(shape=confusion_matrix_test.shape)
    for curr_idx in range(confusion_matrix_test.shape[0]):
        confusion_matrix_obj['z_list'].append([])
        for curr_trial_idx in range(confusion_matrix_test.shape[1]):
            #
            curr_confusion_matrix_test = confusion_matrix_test[curr_idx, curr_trial_idx]
            confusion_matrix_obj['z_list'][curr_idx].append(curr_confusion_matrix_test.tolist())
            #
            if curr_confusion_matrix_test.size == 1:
                curr_positive_percent = 0
                curr_positive_abs = 0
            else:
                curr_positive_percent = curr_confusion_matrix_test[1].sum()/float(curr_confusion_matrix_test.sum())
                curr_positive_abs = curr_confusion_matrix_test[1].sum()
            positive_label_percent[curr_idx, curr_trial_idx] = curr_positive_percent
            positive_label_abs[curr_idx, curr_trial_idx] = curr_positive_abs
    positive_label_percent_obj['list'] = positive_label_percent.tolist()
    positive_label_abs_obj['list'] = positive_label_abs.tolist()
    #
    results_obj['train_test_amr_mmd_divergence'] = train_test_amr_mmd_divergence.tolist()
    #
    print json.dumps(results_obj, ensure_ascii=True, sort_keys=True, indent=5)
    with open(cap.absolute_path+file_path+'.json', 'w') as f:
        json.dump(results_obj, f, ensure_ascii=True, sort_keys=True, indent=5)


def infer_frm_label_propagation(K, labels_train, train, test):
    labels_test_pred_prob = lp.infer_labels_fr_test(K, None, labels_train, train, test)
    #
    n = labels_test_pred_prob.shape[0]
    assert labels_test_pred_prob.shape[1] == 3
    r = np.random.uniform(size=n)
    # p = labels_test_pred_prob[:, 1] + labels_test_pred_prob[:, 2]
    # q = labels_test_pred_prob[:, 2]/p
    # z1 = np.zeros(n)
    # z1[r <= p] = 1
    # del p
    # z = np.copy(z1)
    # z[r <= q] = 2
    # del q, r
    # z *= z1
    # del z1
    # labels_test_pred = z
    # del z
    labels_test_pred = np.zeros(n)
    labels_test_pred[r <= labels_test_pred_prob[:, 1]] = 1
    return labels_test_pred, labels_test_pred_prob

# def validate(K, labels, algo_name_fr_plots, interaction_count_map=None, interaction_idx_list_map=None):
#     raise NotImplementedError
#     # assuming three types of labels in order: 0-false, 1-true, 2-swap (close to truth)
#     print 'labels', labels
#     assert K.shape[0] == K.shape[1] and len(K.shape) == 2
#     assert (interaction_count_map is None and interaction_idx_list_map is None) \
#            or (interaction_count_map is not None and interaction_idx_list_map is not None)
#     n = K.shape[0]
#     print 'n', n
#     #
#     adj_rand_score_test = -1*np.ones(num_trials)
#     positive_label_precision_test = -1*np.ones(num_trials)
#     positive_label_recall_test = -1*np.ones(num_trials)
#     positive_label_f1_score_test = -1*np.ones(num_trials)
#     confusion_matrix_test = np.empty(shape=num_trials, dtype=np.object)
#     positive_label_precision_recall_curve_map = np.empty(shape=num_trials, dtype=np.object)
#     #
#     if interaction_count_map is not None and interaction_idx_list_map is not None:
#         is_max_lkl_label = True
#         max_lkl_adj_rand_score_test = -1*np.ones(num_trials)
#         max_lkl_positive_label_precision_test = -1*np.ones(num_trials)
#         max_lkl_positive_label_recall_test = -1*np.ones(num_trials)
#         max_lkl_positive_label_f1_score_test = -1*np.ones(num_trials)
#         max_lkl_confusion_matrix_test = np.empty(shape=num_trials, dtype=np.object)
#         max_lkl_positive_label_precision_recall_curve_map = np.empty(shape=num_trials, dtype=np.object)
#     else:
#         is_max_lkl_label = False
#     #
#     training_frac = 0.8
#     print 'Learning and Inferring for {} runs.'.format(num_trials)
#     for i in range(num_trials):
#         sys.stdout.write('.')
#         sys.stdout.flush()
#         num_test_samples = round(n*(1-training_frac))
#         test_start_idx = rnd.randint(n*training_frac)
#         print 'test_start_idx', test_start_idx
#         all_idx = np.arange(n)
#         test = all_idx[test_start_idx:test_start_idx+num_test_samples]
#         print 'test', test
#         if debug:
#             print 'test', test
#         train = np.setdiff1d(np.arange(0, n), test)
#         print 'train', train
#         labels_train = labels[train]
#         labels_test = labels[test]
#         #
#         if classifier == 'svm':
#             K_train = K[np.meshgrid(train, train, indexing='ij')]
#             K_test = K[np.meshgrid(test, train, indexing='ij')]
#             labels_test_pred, labels_test_pred_prob = infer_nd_lrn_svm(K_train, labels_train, K_test)
#         elif classifier == 'lp':
#             labels_test_pred, labels_test_pred_prob = infer_frm_label_propagation(K, labels_train, train, test)
#         else:
#             raise NotImplementedError
#         # inferring metrics
#         adj_rand_score_test[i], confusion_matrix_test[i], positive_label_precision_test[i], \
#         positive_label_recall_test[i], positive_label_f1_score_test[i], positive_label_precision_recall_curve_map[i] \
#             = get_metrics(labels_test, labels_test_pred, labels_test_pred_prob[:, 1])
#         #
#         if is_max_lkl_label:
#             max_lkl_labels_test, max_lkl_labels_test_pred, max_lkl_labels_test_pred_prob_fr_positive_label \
#                  = get_max_lkl_inferred_nd_true_labels_fr_interaction(test, labels_test_pred, labels_test_pred_prob[:, 1],
#                                                        interaction_count_map, interaction_idx_list_map)
#             max_lkl_adj_rand_score_test[i],\
#             max_lkl_confusion_matrix_test[i],\
#             max_lkl_positive_label_precision_test[i], \
#             max_lkl_positive_label_recall_test[i],\
#             max_lkl_positive_label_f1_score_test[i],\
#             max_lkl_positive_label_precision_recall_curve_map[i] \
#                 = get_metrics(max_lkl_labels_test, max_lkl_labels_test_pred, max_lkl_labels_test_pred_prob_fr_positive_label)
#         #
#         if debug:
#             print 'adj_rand_score_test[i]', adj_rand_score_test[i]
#     #
#     print_nd_plot_metric_summary(
#         num_trials,
#         algo_name_fr_plots,
#         adj_rand_score_test,
#         confusion_matrix_test,
#         positive_label_precision_test,
#         positive_label_recall_test,
#         positive_label_f1_score_test,
#         positive_label_precision_recall_curve_map
#     )
#     #
#     if is_max_lkl_label:
#         print_nd_plot_metric_summary(
#             num_trials,
#             algo_name_fr_plots,
#             max_lkl_adj_rand_score_test,
#             max_lkl_confusion_matrix_test,
#             max_lkl_positive_label_precision_test,
#             max_lkl_positive_label_recall_test,
#             max_lkl_positive_label_f1_score_test,
#             max_lkl_positive_label_precision_recall_curve_map,
#             is_max_lkl_label=is_max_lkl_label
#         )


def get_idx_fr_paper_set(amr_graphs, curr_amr_set, is_list):
    idx_list = []
    n = amr_graphs.shape[0]
    #
    if is_list:
        assert isinstance(curr_amr_set, list)
    else:
        assert not isinstance(curr_amr_set, list)
    #
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs[curr_idx, 0]
        curr_path = curr_amr_graph_map['path']
        #
        if is_list:
            for curr_paper in curr_amr_set:
                if curr_paper in curr_path:
                    assert curr_idx not in idx_list
                    idx_list.append(curr_idx)
        else:
            if curr_amr_set in curr_path:
                assert curr_idx not in idx_list
                idx_list.append(curr_idx)
    idx_list = np.array(idx_list)
    idx_list.sort()
    return idx_list


def random_subset_indices_at_interaction_level(amr_graphs_subset, labels_subset, subset_idx, is_dep_only, frac):
    raise DeprecationWarning
    print 'subset_idx', subset_idx
    #
    if is_dep_only:
        _, interaction_idx_list_map, _ \
            = edk.get_statistics_dependency_graphs_fr_interaction(amr_graphs_subset, labels_subset)
    else:
        _, interaction_idx_list_map, _ \
            = edk.get_statistics_amr_graphs_fr_interaction(amr_graphs_subset, labels_subset)
    interactions_list = interaction_idx_list_map.keys()
    print 'interactions_list', interactions_list
    #
    n = len(interactions_list)
    print 'n', n
    num_subset = int(round(n*frac))
    print 'num_subset', num_subset
    interaction_subset = r.sample(interactions_list, num_subset)
    print 'interaction_subset', interaction_subset
    interactions_list = None
    #
    random_subset_idx_in_org_subset = []
    for curr_interaction in interaction_subset:
        curr_idx_list = interaction_idx_list_map[curr_interaction]
        print 'curr_idx_list', curr_idx_list
        random_subset_idx_in_org_subset += curr_idx_list
    print 'random_subset_idx_in_org_subset', random_subset_idx_in_org_subset
    interaction_subset = None
    curr_interaction = None
    #
    random_subset_idx = subset_idx[random_subset_idx_in_org_subset]
    random_subset_idx_in_org_subset = None
    assert not isinstance(random_subset_idx, list), 'should be an array'
    random_subset_idx.sort()
    print 'random_subset_idx', random_subset_idx
    return random_subset_idx


def eval_over_fitting_of_test_sets_wrt_train(K, amr_graphs, is_train_fixed):
    raise DeprecationWarning
    if is_training_fixed:
        num_amr_sets = len(amr_sets.amr_sets_list_auto_merged)
    else:
        num_amr_sets = len(amr_sets.amr_sets_list_auto)
    #
    n = amr_graphs.shape[0]
    assert amr_graphs.shape[1] == 1
    assert n == K.shape[0] == K.shape[1]
    #
    divergence_map = {}
    divergence_map['kl_knn'] = []
    divergence_map['kl_kd'] = []
    divergence_map['mmd'] = []
    for i in range(num_amr_sets):
        sys.stdout.write('.')
        sys.stdout.flush()
        if is_training_fixed:
            curr_paper_set = amr_sets.amr_sets_list_auto_merged[i]
        else:
            curr_paper_set = amr_sets.amr_sets_list_auto[i]
        test = get_idx_fr_paper_set(amr_graphs, curr_paper_set, is_list=(not is_training_fixed))
        print '({},{})'.format(curr_paper_set, len(test))
        if is_training_fixed:
            train = get_idx_fr_paper_set(amr_graphs, amr_sets.amr_sets_list_gold, is_list=True)
        else:
            train = np.setdiff1d(np.arange(0, n), test)
        print 'train', train
        Kii = K[np.meshgrid(test.astype(np.int32), test.astype(np.int32), indexing='ij', sparse=True, copy=False)]
        Kjj = K[np.meshgrid(train.astype(np.int32), train.astype(np.int32), indexing='ij', sparse=True, copy=False)]
        Kij = K[np.meshgrid(test.astype(np.int32), train.astype(np.int32), indexing='ij', sparse=True, copy=False)]
        #
        divergence_map['kl_knn'].append(edk.eval_divergence(Kii=Kii, Kjj=Kjj, Kij=Kij, algo='kl_knn', nn=5))
        print divergence_map['kl_knn']
        divergence_map['kl_kd'].append(edk.eval_divergence(Kii=Kii, Kjj=Kjj, Kij=Kij, algo='kl_kd'))
        print divergence_map['kl_kd']
        divergence_map['mmd'].append(edk.eval_divergence(Kii=Kii, Kjj=Kjj, Kij=Kij, algo='mmd'))
        print divergence_map['mmd']
    #
    file_path = './validation_plots_nd_text_output/test_train_div'
    if is_training_fixed:
        file_path += '_train_fixed'
    with open(cap.absolute_path+file_path+'.json', 'w') as f:
        json.dump(divergence_map, f, ensure_ascii=True, sort_keys=True, indent=5)


class ValidatePaperSets:
    def __init__(self, K, amr_graphs, interaction_idx_list_map, filtered_interaction_idx_list=None, is_alternative_data=False):
        self.K = K
        self.amr_graphs = amr_graphs
        self.interaction_idx_list_map = interaction_idx_list_map
        if isinstance(filtered_interaction_idx_list, list):
            self.filtered_interaction_idx_list = np.array(filtered_interaction_idx_list)
        else:
            self.filtered_interaction_idx_list = filtered_interaction_idx_list
        #
        self.is_alternative_data = is_alternative_data
        if self.is_alternative_data:
            self.amr_sets_fr_test = amr_sets_aimed.amr_sets_list
            self.amr_papers_fr_test = amr_sets_aimed.amr_sets_list_merged
        else:
            self.amr_sets_fr_test = amr_sets.amr_sets_list_auto
            self.amr_papers_fr_test = amr_sets.amr_sets_list_auto_merged
        #
        num_amrs = self.amr_graphs.shape[0]
        self.num_amrs = num_amrs
        assert self.amr_graphs.shape[1] == 1
        assert self.amr_graphs.shape[0] == num_amrs
        assert self.K.shape[0] == self.K.shape[1] == num_amrs, 'K must be kernel matrix not divergence matrix'
        #
        self.num_interactions = len(interaction_idx_list_map)
        #
        self.num_trials = 25
        self.fraction_subset_default = 0.8
        self.num_amr_sets_per_train_trial = None
        self.max_num_trials = 25
        #
        self.train_interaction_idx_random = None
        self.test_interaction_idx_random = None
        self.train_amr_idx_random = None
        self.test_amr_idx_random = None
        self.train_test_amr_mmd_divergence = None
        #
        self.amr_idx_to_interaction_idx = -1*np.ones(num_amrs, dtype=np.int)
        interactions_list = self.interaction_idx_list_map.keys()
        for curr_interaction_idx in range(len(interactions_list)):
            curr_interaction = interactions_list[curr_interaction_idx]
            amr_idx_in_curr_interaction = self.interaction_idx_list_map[curr_interaction]
            self.amr_idx_to_interaction_idx[amr_idx_in_curr_interaction] = curr_interaction_idx
        assert np.all(self.amr_idx_to_interaction_idx >= 0)
        print 'self.amr_idx_to_interaction_idx', self.amr_idx_to_interaction_idx

    def random_subset_indices(self, org_idx, fraction_subset_default=None):
        if debug:
            print 'org_idx', org_idx
        #
        if fraction_subset_default is None:
            fraction_subset_default = self.fraction_subset_default
        #
        n = len(org_idx)
        num_subset = int(round(n*fraction_subset_default))
        subset_idx = r.sample(org_idx, num_subset)
        org_idx = None
        subset_idx = np.array(subset_idx)
        subset_idx.sort()
        if debug:
            print 'subset_idx', subset_idx
        return subset_idx

    def get_amr_idx_frm_interaction_idx_list(self, interaction_idx_list):
        interaction_keys_list = self.interaction_idx_list_map.keys()
        #
        amr_idx_list = []
        for curr_interaction_idx in interaction_idx_list:
            curr_amr_idx_in_interaction = self.interaction_idx_list_map[interaction_keys_list[curr_interaction_idx]]
            amr_idx_list += curr_amr_idx_in_interaction
        amr_idx_list = np.array(amr_idx_list)
        amr_idx_list.sort()
        return amr_idx_list

    def get_idx_interaction_frm_kernel_idx_list(self, org_sel_idx_list):
        interaction_keys_list = self.interaction_idx_list_map.keys()
        #
        new_idx_list = []
        n = len(interaction_keys_list)
        for i in range(n):
            i_idx_list = self.interaction_idx_list_map[interaction_keys_list[i]]
            count = 0
            for curr_idx in i_idx_list:
                if curr_idx in org_sel_idx_list:
                    count += 1
            if count == len(i_idx_list):
                new_idx_list.append(i)
            elif count == 0:
                pass
            else:
                print 'count', count
                print 'i_idx_list', i_idx_list
                print 'org_sel_idx_list', org_sel_idx_list
                print 'warning: interaction from more than one amr subsets'
                raise AssertionError
        new_idx_list = np.array(new_idx_list)
        new_idx_list.sort()
        return new_idx_list

    def sample_train_test_random_subset(self):
        # assuming three types of labels in order: 0-false, 1-true, 2-swap (close to truth)
        num_amr_sets = len(self.amr_sets_fr_test)
        print 'num_amr_sets', num_amr_sets
        #
        train_interaction_idx_random = np.empty(shape=(1,self.num_trials), dtype=np.object)
        test_interaction_idx_random = np.empty(shape=(1,self.num_trials), dtype=np.object)
        #
        train_amr_idx_random = np.empty(shape=(1,self.num_trials), dtype=np.object)
        test_amr_idx_random = np.empty(shape=(1,self.num_trials), dtype=np.object)
        #
        train_test_amr_mmd_divergence = -1*np.ones((1,self.num_trials))
        #
        curr_paper_set = self.amr_papers_fr_test
        #
        # we are explicitly considering interaction level indices here
        #
        test_amr_idx = get_idx_fr_paper_set(self.amr_graphs, curr_paper_set, is_list=True)
        test_interaction_idx = self.get_idx_interaction_frm_kernel_idx_list(test_amr_idx)
        print '({},{})'.format(curr_paper_set, len(test_amr_idx))
        print '({},{})'.format(curr_paper_set, len(test_interaction_idx))
        #
        train_interaction_idx = np.setdiff1d(np.arange(0, self.num_interactions), test_interaction_idx)
        #
        for curr_trial_idx in range(self.num_trials):
            train_interaction_idx_random[0,curr_trial_idx] = self.random_subset_indices(train_interaction_idx)
            test_interaction_idx_random[0,curr_trial_idx] = self.random_subset_indices(test_interaction_idx, fraction_subset_default=0.1)
            #
            train_amr_idx_random[0,curr_trial_idx] =\
                self.get_amr_idx_frm_interaction_idx_list(train_interaction_idx_random[0,curr_trial_idx])
            test_amr_idx_random[0,curr_trial_idx] =\
                self.get_amr_idx_frm_interaction_idx_list(test_interaction_idx_random[0,curr_trial_idx])
            #
            i = train_amr_idx_random[0,curr_trial_idx]
            j = test_amr_idx_random[0,curr_trial_idx]
            Kii = self.K[np.meshgrid(i.astype(np.int32), i.astype(np.int32), indexing='ij', sparse=True, copy=False)]
            Kjj = self.K[np.meshgrid(j.astype(np.int32), j.astype(np.int32), indexing='ij', sparse=True, copy=False)]
            Kij = self.K[np.meshgrid(i.astype(np.int32), j.astype(np.int32), indexing='ij', sparse=True, copy=False)]
            #
            train_test_amr_mmd_divergence[0,curr_trial_idx] =\
                edk.eval_max_mean_discrepancy(Kii=Kii, Kjj=Kjj, Kij=Kij)
            #
            print '(ttd: {})'.format(train_test_amr_mmd_divergence[0,curr_trial_idx])
        #
        self.train_interaction_idx_random = train_interaction_idx_random
        self.test_interaction_idx_random = test_interaction_idx_random
        self.train_amr_idx_random = train_amr_idx_random
        self.test_amr_idx_random = test_amr_idx_random
        self.train_test_amr_mmd_divergence = train_test_amr_mmd_divergence
        #
        if not ch.is_hpcc:
            print train_test_amr_mmd_divergence
            train_test_amr_mmd_divergence = train_test_amr_mmd_divergence.flatten()
            print train_test_amr_mmd_divergence
            train_test_amr_mmd_divergence.sort()
            print train_test_amr_mmd_divergence
            plt.close()
            plt.plot(train_test_amr_mmd_divergence, 'kx')
            plt.savefig('./validation_plots_nd_text_output/summary/train_test_amr_mmd_divergence.pdf', dpi=300, format='pdf')
            plt.close()

    def sample_test_random_subset_rest_train(self):
        # assuming three types of labels in order: 0-false, 1-true, 2-swap (close to truth)
        num_amr_sets = len(self.amr_sets_fr_test)
        print 'num_amr_sets', num_amr_sets
        #
        train_interaction_idx_random = np.empty(shape=(1,self.num_trials), dtype=np.object)
        test_interaction_idx_random = np.empty(shape=(1,self.num_trials), dtype=np.object)
        #
        train_amr_idx_random = np.empty(shape=(1,self.num_trials), dtype=np.object)
        test_amr_idx_random = np.empty(shape=(1,self.num_trials), dtype=np.object)
        #
        train_test_amr_mmd_divergence = -1*np.ones((1,self.num_trials))
        #
        curr_paper_set = self.amr_papers_fr_test
        #
        # we are explicitly considering interaction level indices here
        #
        test_amr_idx = get_idx_fr_paper_set(self.amr_graphs, curr_paper_set, is_list=True)
        test_interaction_idx = self.get_idx_interaction_frm_kernel_idx_list(test_amr_idx)
        print '({},{})'.format(curr_paper_set, len(test_amr_idx))
        print '({},{})'.format(curr_paper_set, len(test_interaction_idx))
        #
        # train_interaction_idx = np.setdiff1d(np.arange(0, self.num_interactions), test_interaction_idx)
        #
        for curr_trial_idx in range(self.num_trials):
            test_interaction_idx_random[0,curr_trial_idx] = self.random_subset_indices(test_interaction_idx, fraction_subset_default=0.1)
            train_interaction_idx = np.setdiff1d(np.arange(0, self.num_interactions), test_interaction_idx_random[0,curr_trial_idx])
            train_interaction_idx_random[0,curr_trial_idx] = train_interaction_idx
            #
            train_amr_idx_random[0,curr_trial_idx] =\
                self.get_amr_idx_frm_interaction_idx_list(train_interaction_idx_random[0,curr_trial_idx])
            test_amr_idx_random[0,curr_trial_idx] =\
                self.get_amr_idx_frm_interaction_idx_list(test_interaction_idx_random[0,curr_trial_idx])
            #
            i = train_amr_idx_random[0,curr_trial_idx]
            j = test_amr_idx_random[0,curr_trial_idx]
            Kii = self.K[np.meshgrid(i.astype(np.int32), i.astype(np.int32), indexing='ij', sparse=True, copy=False)]
            Kjj = self.K[np.meshgrid(j.astype(np.int32), j.astype(np.int32), indexing='ij', sparse=True, copy=False)]
            Kij = self.K[np.meshgrid(i.astype(np.int32), j.astype(np.int32), indexing='ij', sparse=True, copy=False)]
            #
            train_test_amr_mmd_divergence[0,curr_trial_idx] =\
                edk.eval_max_mean_discrepancy(Kii=Kii, Kjj=Kjj, Kij=Kij)
            #
            print '(ttd: {})'.format(train_test_amr_mmd_divergence[0,curr_trial_idx])
        #
        self.train_interaction_idx_random = train_interaction_idx_random
        self.test_interaction_idx_random = test_interaction_idx_random
        self.train_amr_idx_random = train_amr_idx_random
        self.test_amr_idx_random = test_amr_idx_random
        self.train_test_amr_mmd_divergence = train_test_amr_mmd_divergence
        #
        if not ch.is_hpcc:
            print train_test_amr_mmd_divergence
            train_test_amr_mmd_divergence = train_test_amr_mmd_divergence.flatten()
            print train_test_amr_mmd_divergence
            train_test_amr_mmd_divergence.sort()
            print train_test_amr_mmd_divergence
            plt.close()
            plt.plot(train_test_amr_mmd_divergence, 'kx')
            plt.savefig('./validation_plots_nd_text_output/summary/train_test_amr_mmd_divergence.pdf', dpi=300, format='pdf')
            plt.close()

    def filter_interaction_idx(self, interaction_idx_list):
        if self.filtered_interaction_idx_list is not None:
            interaction_idx_list = np.intersect1d(interaction_idx_list, self.filtered_interaction_idx_list)
        return interaction_idx_list

    def sample_train_test_interaction_random_subset_at_paper_level(self):
        # assuming three types of labels in order: 0-false, 1-true, 2-swap (close to truth)
        num_amr_sets = len(self.amr_sets_fr_test)
        print 'num_amr_sets', num_amr_sets
        #
        train_interaction_idx_random = np.empty(shape=(num_amr_sets, self.num_trials), dtype=np.object)
        test_interaction_idx_random = np.empty(shape=(num_amr_sets, self.num_trials), dtype=np.object)
        #
        train_amr_idx_random = np.empty(shape=(num_amr_sets, self.num_trials), dtype=np.object)
        test_amr_idx_random = np.empty(shape=(num_amr_sets, self.num_trials), dtype=np.object)
        #
        train_test_amr_mmd_divergence = -1*np.ones((num_amr_sets, self.num_trials))
        #
        for curr_amr_set_idx in range(num_amr_sets):
            sys.stdout.write('.')
            sys.stdout.flush()
            #
            curr_paper_set = self.amr_sets_fr_test[curr_amr_set_idx]
            #
            # we are explicitly considering interaction level indices here
            test_amr_idx = get_idx_fr_paper_set(self.amr_graphs, curr_paper_set, is_list=True)
            test_interaction_idx = self.get_idx_interaction_frm_kernel_idx_list(test_amr_idx)
            test_interaction_idx = self.filter_interaction_idx(test_interaction_idx)
            print '({},{})'.format(curr_paper_set, len(test_amr_idx))
            print '({},{})'.format(curr_paper_set, len(test_interaction_idx))
            #
            train_interaction_idx = np.setdiff1d(np.arange(0, self.num_interactions), test_interaction_idx)
            train_interaction_idx = self.filter_interaction_idx(train_interaction_idx)
            #
            for curr_trial_idx in range(self.num_trials):
                train_interaction_idx_random[curr_amr_set_idx, curr_trial_idx] = self.random_subset_indices(train_interaction_idx)
                test_interaction_idx_random[curr_amr_set_idx, curr_trial_idx] = self.random_subset_indices(test_interaction_idx)
                #
                train_amr_idx_random[curr_amr_set_idx, curr_trial_idx] =\
                    self.get_amr_idx_frm_interaction_idx_list(train_interaction_idx_random[curr_amr_set_idx, curr_trial_idx])
                test_amr_idx_random[curr_amr_set_idx, curr_trial_idx] =\
                    self.get_amr_idx_frm_interaction_idx_list(test_interaction_idx_random[curr_amr_set_idx, curr_trial_idx])
                #
                i = train_amr_idx_random[curr_amr_set_idx, curr_trial_idx]
                j = test_amr_idx_random[curr_amr_set_idx, curr_trial_idx]
                Kii = self.K[np.meshgrid(i.astype(np.int32), i.astype(np.int32), indexing='ij', sparse=True, copy=False)]
                Kjj = self.K[np.meshgrid(j.astype(np.int32), j.astype(np.int32), indexing='ij', sparse=True, copy=False)]
                Kij = self.K[np.meshgrid(i.astype(np.int32), j.astype(np.int32), indexing='ij', sparse=True, copy=False)]
                #
                train_test_amr_mmd_divergence[curr_amr_set_idx, curr_trial_idx] =\
                    edk.eval_max_mean_discrepancy(Kii=Kii, Kjj=Kjj, Kij=Kij)
                #
                print '(ttd: {})'.format(train_test_amr_mmd_divergence[curr_amr_set_idx, curr_trial_idx])
        #
        self.train_interaction_idx_random = train_interaction_idx_random
        self.test_interaction_idx_random = test_interaction_idx_random
        self.train_amr_idx_random = train_amr_idx_random
        self.test_amr_idx_random = test_amr_idx_random
        self.train_test_amr_mmd_divergence = train_test_amr_mmd_divergence
        #
        if not ch.is_hpcc:
            print train_test_amr_mmd_divergence
            train_test_amr_mmd_divergence = train_test_amr_mmd_divergence.flatten()
            print train_test_amr_mmd_divergence
            train_test_amr_mmd_divergence.sort()
            print train_test_amr_mmd_divergence
            plt.close()
            plt.plot(train_test_amr_mmd_divergence, 'kx')
            plt.savefig('./validation_plots_nd_text_output/summary/train_test_amr_mmd_divergence.pdf', dpi=300, format='pdf')
            plt.close()

    def choose_train_amr_sets(self):
        # assuming three types of labels in order: 0-false, 1-true, 2-swap (close to truth)
        num_amr_sets = len(self.amr_sets_fr_test)
        print 'num_amr_sets', num_amr_sets
        #
        # num_amr_sets_per_train_trial = int(round(self.fraction_subset_default*(num_amr_sets-1)))
        # print 'num_amr_sets_per_train_trial', num_amr_sets_per_train_trial
        num_combinations = m.factorial(num_amr_sets-1)/(m.factorial(self.num_amr_sets_per_train_trial)*m.factorial(num_amr_sets-1-self.num_amr_sets_per_train_trial))
        if num_combinations <= self.max_num_trials:
            self.num_trials = num_combinations
        else:
            self.num_trials = self.max_num_trials
        # assert num_max_trials >= self.num_trials
        #
        train_interaction_idx_random = np.empty(shape=(num_amr_sets, self.num_trials), dtype=np.object)
        test_interaction_idx_random = np.empty(shape=(num_amr_sets, self.num_trials), dtype=np.object)
        #
        train_amr_idx_random = np.empty(shape=(num_amr_sets, self.num_trials), dtype=np.object)
        test_amr_idx_random = np.empty(shape=(num_amr_sets, self.num_trials), dtype=np.object)
        #
        train_test_amr_mmd_divergence = -1*np.ones((num_amr_sets, self.num_trials))
        #
        train_amr_sets_idx_combinations_list = []
        #
        amr_sets_idx_range = np.arange(num_amr_sets)
        for curr_test_amr_set_idx in amr_sets_idx_range:
            sys.stdout.write('.')
            sys.stdout.flush()
            #
            curr_test_paper_set = self.amr_sets_fr_test[curr_test_amr_set_idx]
            #
            curr_train_amr_sets_idx = np.setdiff1d(amr_sets_idx_range, np.array([curr_test_amr_set_idx]))
            #
            # we are explicitly considering interaction level indices here
            test_amrs_idx = get_idx_fr_paper_set(self.amr_graphs, curr_test_paper_set, is_list=True)
            test_interactions_idx = self.get_idx_interaction_frm_kernel_idx_list(test_amrs_idx)
            #
            print '({},{})'.format(curr_test_paper_set, len(test_amrs_idx))
            print '({},{})'.format(curr_test_paper_set, len(test_interactions_idx))
            #
            train_amr_sets_idx_combinations = \
                [list(curr_combination) for curr_combination in it.combinations(curr_train_amr_sets_idx, self.num_amr_sets_per_train_trial)]
            print 'train_amr_sets_idx_combinations', train_amr_sets_idx_combinations
            if self.num_trials < num_combinations:
                train_amr_sets_idx_combinations = r.sample(train_amr_sets_idx_combinations, self.num_trials)
            #
            train_amr_sets_idx_combinations_list.append(train_amr_sets_idx_combinations)
            # print 'train_amr_sets_idx_combinations after random sub-selection', train_amr_sets_idx_combinations
            curr_trial_idx = -1
            for curr_train_amr_sets_idx_combination in train_amr_sets_idx_combinations:
                curr_trial_idx += 1
                #
                curr_paper_set_train  = []
                for curr_train_amr_set_idx_in_comb in curr_train_amr_sets_idx_combination:
                    curr_paper_set_train += self.amr_sets_fr_test[curr_train_amr_set_idx_in_comb]
                curr_paper_set_train += amr_sets.amr_sets_list_gold
                curr_train_amr_sets_idx_combination = None
                curr_train_amr_set_idx_in_comb = None
                #
                curr_train_amr_idx = get_idx_fr_paper_set(self.amr_graphs, curr_paper_set_train, is_list=True)
                curr_train_interaction_idx = self.get_idx_interaction_frm_kernel_idx_list(curr_train_amr_idx)
                curr_paper_set_train = None
                #
                train_interaction_idx_random[curr_test_amr_set_idx, curr_trial_idx] = curr_train_interaction_idx
                train_amr_idx_random[curr_test_amr_set_idx, curr_trial_idx] = curr_train_amr_idx
                #
                test_interaction_idx_random[curr_test_amr_set_idx, curr_trial_idx] = test_interactions_idx
                test_amr_idx_random[curr_test_amr_set_idx, curr_trial_idx] = test_amrs_idx
                #
                i = train_amr_idx_random[curr_test_amr_set_idx, curr_trial_idx]
                j = test_amr_idx_random[curr_test_amr_set_idx, curr_trial_idx]
                Kii = self.K[np.meshgrid(i.astype(np.int32), i.astype(np.int32), indexing='ij', sparse=True, copy=False)]
                Kjj = self.K[np.meshgrid(j.astype(np.int32), j.astype(np.int32), indexing='ij', sparse=True, copy=False)]
                Kij = self.K[np.meshgrid(i.astype(np.int32), j.astype(np.int32), indexing='ij', sparse=True, copy=False)]
                #
                train_test_amr_mmd_divergence[curr_test_amr_set_idx, curr_trial_idx] =\
                    edk.eval_max_mean_discrepancy(Kii=Kii, Kjj=Kjj, Kij=Kij)
                print '(ttd: {})'.format(train_test_amr_mmd_divergence[curr_test_amr_set_idx, curr_trial_idx])
        #
        self.train_interaction_idx_random = train_interaction_idx_random
        self.test_interaction_idx_random = test_interaction_idx_random
        self.train_amr_idx_random = train_amr_idx_random
        self.test_amr_idx_random = test_amr_idx_random
        self.train_amr_sets_idx_combinations_list = train_amr_sets_idx_combinations_list
        self.train_test_amr_mmd_divergence = train_test_amr_mmd_divergence
        print 'self.train_test_amr_mmd_divergence ', self.train_test_amr_mmd_divergence
        print 'self.train_test_amr_mmd_divergence.mean()', self.train_test_amr_mmd_divergence.mean()
        print 'self.train_test_amr_mmd_divergence.std()', self.train_test_amr_mmd_divergence.std()
        #
        with open(cap.absolute_path+'./train_test_amr_mmd_divergence.json', 'w') as f:
            json.dump(self.train_test_amr_mmd_divergence.tolist(), f, ensure_ascii=True, sort_keys=True, indent=5)
        #
        if not ch.is_hpcc:
            print train_test_amr_mmd_divergence
            train_test_amr_mmd_divergence = train_test_amr_mmd_divergence.flatten()
            print train_test_amr_mmd_divergence
            train_test_amr_mmd_divergence.sort()
            print train_test_amr_mmd_divergence
            plt.close()
            plt.plot(train_test_amr_mmd_divergence, 'kx')
            plt.savefig('./validation_plots_nd_text_output/summary/train_test_amr_mmd_divergence.pdf', dpi=300, format='pdf')
            plt.close()

    def get_max_lkl_inferred_nd_true_labels_fr_interaction(self, test_idx, labels_inferred, labels_inferred_prob_fr_positive):
        n = test_idx.size
        print 'test_idx', test_idx
        true_labels_list = []
        inferred_labels_list_fr_max_lkl = []
        inferred_labels_max_lkl_list = []
        #
        print 'n', n
        print 'len(interaction_idx_list_map)', len(self.interaction_idx_list_map)
        #
        test_interaction_idx = self.amr_idx_to_interaction_idx[test_idx]
        print 'test_interaction_idx', test_interaction_idx
        test_interaction_idx = np.unique(test_interaction_idx)
        print 'test_interaction_idx', test_interaction_idx
        #
        interactions_list = self.interaction_idx_list_map.keys()
        # interactions_list = np.array(interactions_list)
        print 'interactions_list', interactions_list
        #
        test_interaction_list = []
        for curr_test_interaction_idx in test_interaction_idx:
            test_interaction_list.append(interactions_list[curr_test_interaction_idx])
        print 'test_interaction_list', test_interaction_list
        #
        for curr_interaction_tuple in test_interaction_list:
            print '******************************************'
            print 'curr_interaction_tuple', curr_interaction_tuple
            curr_true_label = curr_interaction_tuple[1]
            print 'curr_true_label', curr_true_label
            curr_interaction_idx_list = np.array(self.interaction_idx_list_map[curr_interaction_tuple])
            print 'curr_interaction_idx_list', curr_interaction_idx_list
            # assuming that indices in test are sorted
            assert all(test_idx[i] <= test_idx[i+1] for i in range(len(test_idx)-1))
            #
            curr_interaction_idx_list = np.intersect1d(test_idx, curr_interaction_idx_list)
            print 'curr_interaction_idx_list', curr_interaction_idx_list
            #
            curr_interaction_idx_list = np.searchsorted(test_idx, curr_interaction_idx_list)
            assert np.all(curr_interaction_idx_list < n)
            print 'curr_interaction_idx_list', curr_interaction_idx_list
            #
            if curr_interaction_idx_list.size == 0:
                continue
            #
            curr_labels_inferred = labels_inferred[curr_interaction_idx_list]
            print 'curr_labels_inferred', curr_labels_inferred
            curr_labels_inferred_prob_fr_positive = labels_inferred_prob_fr_positive[curr_interaction_idx_list]
            print 'curr_labels_inferred_prob_fr_positive', curr_labels_inferred_prob_fr_positive
            curr_label_inferred_max_lkl = curr_labels_inferred[curr_labels_inferred_prob_fr_positive.argmax()]
            print 'curr_label_inferred_max_lkl', curr_label_inferred_max_lkl
            #
            true_labels_list.append(curr_true_label)
            inferred_labels_list_fr_max_lkl.append(curr_label_inferred_max_lkl)
            inferred_labels_max_lkl_list.append(curr_labels_inferred_prob_fr_positive.max())
        #
        true_labels_list = np.array(true_labels_list)
        print 'len(true_labels_list)', len(true_labels_list)
        inferred_labels_list_fr_max_lkl = np.array(inferred_labels_list_fr_max_lkl)
        print 'inferred_labels_list_fr_max_lkl', inferred_labels_list_fr_max_lkl
        inferred_labels_max_lkl_list = np.array(inferred_labels_max_lkl_list)
        print 'inferred_labels_max_lkl_list', inferred_labels_max_lkl_list
        return true_labels_list, inferred_labels_list_fr_max_lkl, inferred_labels_max_lkl_list

    def save(self):
        vps_obj = ValidatePaperSets(None, None, None)
        vps_obj.train_interaction_idx_random = self.train_interaction_idx_random
        vps_obj.test_interaction_idx_random = self.test_interaction_idx_random
        vps_obj.train_amr_idx_random = self.train_amr_idx_random
        vps_obj.test_amr_idx_random = self.test_amr_idx_random
        vps_obj.train_test_amr_mmd_divergence = self.train_test_amr_mmd_divergence
        vps_obj.train_amr_sets_idx_combinations_list = self.train_amr_sets_idx_combinations_list

    def validate_at_paper_level(self, K, labels, algo_name_fr_plots, is_divergence=False):
        # todo: make corresponding new changes in function "eval_over_fitting_of_test_sets_wrt_train" or eliminate the redundancy
        # assuming three types of labels in order: 0-false, 1-true, 2-swap (close to truth)
        assert self.train_interaction_idx_random is not None
        assert self.test_interaction_idx_random is not None
        assert self.train_amr_idx_random is not None
        assert self.test_amr_idx_random is not None
        assert self.train_test_amr_mmd_divergence is not None
        #
        testing_shape = self.train_amr_idx_random.shape
        print 'testing_shape', testing_shape
        assert testing_shape == self.test_amr_idx_random.shape
        assert testing_shape == self.train_interaction_idx_random.shape
        assert testing_shape == self.test_interaction_idx_random.shape
        assert testing_shape == self.train_test_amr_mmd_divergence.shape
        #
        adj_rand_score_test = -1*np.ones(testing_shape)
        positive_label_precision_test = -1*np.ones(testing_shape)
        positive_label_recall_test = -1*np.ones(testing_shape)
        positive_label_f1_score_test = -1*np.ones(testing_shape)
        confusion_matrix_test = np.empty(shape=testing_shape, dtype=np.object)
        positive_label_precision_recall_curve_map = np.empty(shape=testing_shape, dtype=np.object)
        #
        if not is_divergence:
            is_max_lkl_label = True
            max_lkl_adj_rand_score_test = -1*np.ones(testing_shape)
            max_lkl_positive_label_precision_test = -1*np.ones(testing_shape)
            max_lkl_positive_label_recall_test = -1*np.ones(testing_shape)
            max_lkl_positive_label_f1_score_test = -1*np.ones(testing_shape)
            max_lkl_confusion_matrix_test = np.empty(shape=testing_shape, dtype=np.object)
            max_lkl_positive_label_precision_recall_curve_map = np.empty(shape=testing_shape, dtype=np.object)
        else:
            is_max_lkl_label = False
        #
        for curr_amr_set_idx in range(testing_shape[0]):
            sys.stdout.write('.')
            sys.stdout.flush()
            for curr_trial_idx in range(testing_shape[1]):
                if is_divergence:
                    curr_trial_train = self.train_interaction_idx_random[curr_amr_set_idx, curr_trial_idx]
                    curr_trial_test = self.test_interaction_idx_random[curr_amr_set_idx, curr_trial_idx]
                else:
                    curr_trial_train = self.train_amr_idx_random[curr_amr_set_idx, curr_trial_idx]
                    curr_trial_test = self.test_amr_idx_random[curr_amr_set_idx, curr_trial_idx]
                #
                labels_train = labels[curr_trial_train]
                labels_test = labels[curr_trial_test]
                #
                print 'labels_train', labels_train
                print 'labels_test', labels_test
                #
                K_train = K[np.meshgrid(curr_trial_train.astype(np.int32), curr_trial_train.astype(np.int32), indexing='ij', sparse=True, copy=False)]
                print 'Learning SVM classifier ...'
                start_time = time.time()
                svm_clf = skl_svm.SVC(kernel='precomputed', probability=True, verbose=ck.is_svm_verbose, class_weight='auto')
                svm_clf.fit(K_train, labels_train)
                print 'Learning time was ', time.time()-start_time
                print svm_clf.n_support_
                print svm_clf.support_
                print svm_clf.support_vectors_
                #
                print K_train.shape
                curr_trial_train = curr_trial_train[svm_clf.support_]
                labels_train = labels_train[svm_clf.support_]
                K_train = K_train[np.meshgrid(svm_clf.support_.astype(np.int32), svm_clf.support_.astype(np.int32), indexing='ij', sparse=True, copy=False)]
                print 'Re-learning SVM classifier ...'
                start_time = time.time()
                svm_clf = skl_svm.SVC(kernel='precomputed', probability=True, verbose=ck.is_svm_verbose, class_weight='auto')
                svm_clf.fit(K_train, labels_train)
                print 'Re-learning time was ', time.time()-start_time
                print K_train.shape
                print svm_clf.n_support_
                print svm_clf.support_
                print svm_clf.support_vectors_
                #
                K_test = K[np.meshgrid(curr_trial_test.astype(np.int32), curr_trial_train.astype(np.int32), indexing='ij', sparse=True, copy=False)]
                #
                labels_test_pred = svm_clf.predict(K_test)
                labels_test_pred_prob = svm_clf.predict_proba(K_test)
                print 'labels_test_pred', labels_test_pred
                print 'labels_test_pred_prob', labels_test_pred_prob
                #
                # inferring metrics
                print 'labels_test_pred_prob[:, 1]', labels_test_pred_prob[:, 1]
                adj_rand_score_test[curr_amr_set_idx, curr_trial_idx],\
                confusion_matrix_test[curr_amr_set_idx, curr_trial_idx],\
                positive_label_precision_test[curr_amr_set_idx, curr_trial_idx], \
                positive_label_recall_test[curr_amr_set_idx, curr_trial_idx],\
                positive_label_f1_score_test[curr_amr_set_idx, curr_trial_idx],\
                positive_label_precision_recall_curve_map[curr_amr_set_idx, curr_trial_idx] \
                    = get_metrics(labels_test, labels_test_pred, labels_test_pred_prob[:, 1])
                #
                if is_max_lkl_label:
                    max_lkl_labels_test, max_lkl_labels_test_pred, max_lkl_labels_test_pred_prob_fr_positive_label \
                         = self.get_max_lkl_inferred_nd_true_labels_fr_interaction(curr_trial_test, labels_test_pred, labels_test_pred_prob[:, 1])
                    max_lkl_adj_rand_score_test[curr_amr_set_idx, curr_trial_idx],\
                    max_lkl_confusion_matrix_test[curr_amr_set_idx, curr_trial_idx],\
                    max_lkl_positive_label_precision_test[curr_amr_set_idx, curr_trial_idx], \
                    max_lkl_positive_label_recall_test[curr_amr_set_idx, curr_trial_idx],\
                    max_lkl_positive_label_f1_score_test[curr_amr_set_idx, curr_trial_idx],\
                    max_lkl_positive_label_precision_recall_curve_map[curr_amr_set_idx, curr_trial_idx] \
                        = get_metrics(max_lkl_labels_test, max_lkl_labels_test_pred, max_lkl_labels_test_pred_prob_fr_positive_label)
                #
        #
        print_nd_plot_metric_summary(
            algo_name_fr_plots,
            adj_rand_score_test,
            confusion_matrix_test,
            positive_label_precision_test,
            positive_label_recall_test,
            positive_label_f1_score_test,
            positive_label_precision_recall_curve_map,
            self.train_test_amr_mmd_divergence
        )
        #
        if is_max_lkl_label:
            print_nd_plot_metric_summary(
                algo_name_fr_plots,
                max_lkl_adj_rand_score_test,
                max_lkl_confusion_matrix_test,
                max_lkl_positive_label_precision_test,
                max_lkl_positive_label_recall_test,
                max_lkl_positive_label_f1_score_test,
                max_lkl_positive_label_precision_recall_curve_map,
                self.train_test_amr_mmd_divergence,
                is_max_lkl_label=is_max_lkl_label
            )
