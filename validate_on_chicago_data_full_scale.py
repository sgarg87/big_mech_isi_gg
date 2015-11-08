import save_sparse_scipy_matrices as sssm
import numpy as np
import random as r
r.seed(871227)
import validate_on_chicago_data as vcd
import compute_parallel_graph_kernel_matrix_joint_train_data as cpgkmjtd
import train_extractor as te
import semi_automated_extraction_features_chicago_data as saefcd
import eval_divergence_frm_kernel as edk
import constants_absolute_path as cap
import chicago_data as cd
import json


def get_train_kernel_matrix():
    amr_graphs, labels = vcd.get_amr_data()
    #
    n = labels.size
    print 'labels.size', labels.size
    #
    idx_label2 = np.where(labels == 2)
    labels[idx_label2] = 0
    #
    test = vcd.get_chicago_test_data_idx(amr_graphs)
    train = np.setdiff1d(np.arange(0, n), test)
    test = None
    print 'train.shape', train.shape
    #
    k_path = './graph_kernel_matrix_joint_train_data_parallel/num_cores_100.npz'
    K = sssm.load_sparse_csr(k_path)
    print 'K.shape', K.shape
    #
    K_train = K[train, :]
    K_train = K_train.tocsc()
    K_train = K_train[:, train]
    K_train = K_train.tocsr()
    print 'K_train.shape', K_train.shape
    print 'K_train.nnz', K_train.nnz
    return K_train


def eval_inferred_labels(labels_test_pred, amr_graphs_test, chicago_sentence_id__interactions_list_map):
    postive_sample_graphs_idx = np.where(labels_test_pred == 1)
    #
    amr_graphs_test_positive = amr_graphs_test[postive_sample_graphs_idx, :]
    postive_sample_graphs_idx = None
    amr_graphs_test = None
    #
    sentence_id_interactions_list_map \
        = vcd.write_json_fr_interaction_str_sentence_pair_of_graphs(amr_graphs_test_positive, vcd.get_inference_file_path())
    #
    return vcd.evaluate_inference(sentence_id_interactions_list_map, chicago_sentence_id__interactions_list_map)


def evaluate_inferred_interactions_list(inferred_interactions_list, chicago_sentence_id__interactions_list_map):
    print 'len(inferred_interactions_list)', len(inferred_interactions_list)
    num_infer = len(inferred_interactions_list)
    print 'num_infer', num_infer
    #
    num_chicago = 0
    for curr_sentence_id in chicago_sentence_id__interactions_list_map:
        num_chicago += len(chicago_sentence_id__interactions_list_map[curr_sentence_id])
    print 'num_chicago', num_chicago
    curr_sentence_id = None
    #
    match_count = 0
    unique_matched_chicago_interactions = []
    for curr_inferred_interaction_tuple in inferred_interactions_list:
        curr_inferred_interaction = curr_inferred_interaction_tuple[0]
        # print curr_inferred_interaction
        curr_path = curr_inferred_interaction_tuple[1]
        # print curr_path
        curr_inferred_interaction_tuple = None
        curr_sentence_id = saefcd.get_passage_sentence_id_frm_joint_graph_path(curr_path)
        curr_sentence_id = 'ID'+str(curr_sentence_id)
        # print '*********************'
        # print curr_sentence_id
        # print curr_inferred_interaction
        is_match = False
        if curr_sentence_id in chicago_sentence_id__interactions_list_map:
            for curr_chicago_interaction in chicago_sentence_id__interactions_list_map[curr_sentence_id]:
                # print 'curr_chicago_interaction', curr_chicago_interaction
                if saefcd.is_match_interactions(
                        interaction_str_tuple2_chicago=curr_chicago_interaction,
                        interaction_str_tuple1_extracted=curr_inferred_interaction):
                    is_match = True
                    curr_chicago_interaction_tuple = (curr_chicago_interaction, curr_sentence_id)
                    if curr_chicago_interaction_tuple not in unique_matched_chicago_interactions:
                        unique_matched_chicago_interactions.append(curr_chicago_interaction_tuple)
                    break
        #
        if is_match:
            # print 'matched'
            match_count += 1
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


def merge_chicago_nd_inferred(inferred_interactions_list, labels_test_pred_prob, chicago_sentence_id__interactions_list_map):
    new_map = {}
    #
    with open('../chicago_data/stats_id_sentences_map.json', 'r') as g:
        id_sentence_map = json.load(g)
    #
    curr_idx = -1
    #
    for curr_inferred_interaction_tuple in inferred_interactions_list:
        #
        curr_idx += 1
        #
        curr_prob = labels_test_pred_prob[curr_idx]
        #
        curr_inferred_interaction = curr_inferred_interaction_tuple[0]
        curr_inferred_interaction = [curr_inferred_interaction, curr_prob]
        #
        curr_path = curr_inferred_interaction_tuple[1]
        curr_inferred_interaction_tuple = None
        curr_sentence_id = saefcd.get_passage_sentence_id_frm_joint_graph_path(curr_path)
        curr_sentence_id = 'ID'+str(curr_sentence_id)
        #
        curr_sentence_text = id_sentence_map[curr_sentence_id]
        #
        assert curr_sentence_id in chicago_sentence_id__interactions_list_map
        #
        if curr_sentence_id not in new_map:
            new_map[curr_sentence_id] = {}
            new_map[curr_sentence_id]['chicago'] = None
            new_map[curr_sentence_id]['inferred_isi'] = []
            new_map[curr_sentence_id]['text'] = None
        #
        if curr_inferred_interaction not in new_map[curr_sentence_id]['inferred_isi']:
            new_map[curr_sentence_id]['inferred_isi'].append(curr_inferred_interaction)
        #
        if new_map[curr_sentence_id]['chicago'] is None:
            new_map[curr_sentence_id]['chicago'] = chicago_sentence_id__interactions_list_map[curr_sentence_id]
        #
        if new_map[curr_sentence_id]['text'] is None:
            new_map[curr_sentence_id]['text'] = curr_sentence_text
    #
    with open('./chicago_merged_inferred_map.json', 'w') as f:
        json.dump(new_map, f, indent=4)


def filter_inferred_interactions_list(inferred_interactions_list, chicago_sentence_id__interactions_list_map):
    print 'len(inferred_interactions_list)', len(inferred_interactions_list)
    num_infer = len(inferred_interactions_list)
    print 'num_infer', num_infer
    #
    filtered_inferred_interactions_list = []
    #
    for curr_inferred_interaction_tuple in inferred_interactions_list:
        curr_inferred_interaction = curr_inferred_interaction_tuple[0]
        curr_path = curr_inferred_interaction_tuple[1]
        # curr_inferred_interaction_tuple = None
        #
        curr_inferred_interaction_proteins = list(curr_inferred_interaction[1:])
        assert None not in curr_inferred_interaction_proteins
        assert len(curr_inferred_interaction_proteins) == 2
        #
        curr_sentence_id = saefcd.get_passage_sentence_id_frm_joint_graph_path(curr_path)
        curr_sentence_id = 'ID'+str(curr_sentence_id)
        #
        is_match = False
        if curr_sentence_id in chicago_sentence_id__interactions_list_map:
            # if more than one interactions in a sentence, the logic for the comparison below would vary
            assert len(chicago_sentence_id__interactions_list_map[curr_sentence_id]) == 1
            curr_chicago_interaction = chicago_sentence_id__interactions_list_map[curr_sentence_id][0]
            print 'curr_chicago_interaction', curr_chicago_interaction
            #
            is_match = False
            #
            curr_chicago_interaction_proteins = curr_chicago_interaction[1:]
            curr_chicago_interaction = None
            #
            print 'curr_inferred_interaction_proteins', curr_inferred_interaction_proteins
            print 'curr_chicago_interaction_proteins', curr_chicago_interaction_proteins
            #
            if saefcd.match_protein_name(curr_inferred_interaction_proteins[0], curr_chicago_interaction_proteins[0])\
                    and saefcd.match_protein_name(curr_inferred_interaction_proteins[1], curr_chicago_interaction_proteins[1]):
                is_match = True
            elif saefcd.match_protein_name(curr_inferred_interaction_proteins[0], curr_chicago_interaction_proteins[1]) \
                    and saefcd.match_protein_name(curr_inferred_interaction_proteins[1], curr_chicago_interaction_proteins[0]):
                is_match = True
            #
            if is_match:
                filtered_inferred_interactions_list.append(curr_inferred_interaction_tuple)
    #
    print 'len(filtered_inferred_interactions_list)', len(filtered_inferred_interactions_list)
    #
    return filtered_inferred_interactions_list


def get_positive_sentence__interactions_list_map(is_filter=False, is_positive=True):
    sentence_rel_map = vcd.load_json_obj('../chicago_data/stats_dataout.json')
    #
    if is_filter:
        filtered_sentence_ids = cd.get_chicago_filtered_ids_fr_test()
    #
    positive_sentence_rel_map = {}
    #
    for curr_sentence_id in sentence_rel_map:
        if is_filter:
            if curr_sentence_id not in filtered_sentence_ids:
                continue
        #
        curr_tuples_list = sentence_rel_map[curr_sentence_id]
        #
        positive_rel_list = []
        #
        for curr_tuple in curr_tuples_list:
            if curr_tuple[1] == 1 or (not is_positive):
                positive_rel_list.append(tuple(curr_tuple[0]))
        #
        if positive_rel_list:
            positive_sentence_rel_map[curr_sentence_id] = positive_rel_list
    #
    return positive_sentence_rel_map


def filter_map_on_sentence_id(old_map):
    filtered_sentence_ids = cd.get_chicago_filtered_ids_fr_test()
    #
    new_map = {}
    for curr_sentence_id in old_map:
        if curr_sentence_id not in filtered_sentence_ids:
            continue
        else:
            new_map[curr_sentence_id] = old_map[curr_sentence_id]
    return new_map


def get_list_of_proteins_in_extracted_interactions_str_tuples(extracted_interactions_list):
    sentence_id_proteins_list_map = {}
    #
    for curr_interaction in extracted_interactions_list:
        curr_path = curr_interaction[1]
        #
        curr_sentence_id = saefcd.get_passage_sentence_id_frm_joint_graph_path(curr_path)
        curr_path = None
        curr_sentence_id = 'ID'+str(curr_sentence_id)
        #
        print 'curr_interaction', curr_interaction
        #
        if isinstance(curr_interaction[0], tuple):
            curr_interaction[0] = list(curr_interaction[0])
        #
        curr_proteins_list = curr_interaction[0][1:]
        print 'curr_proteins_list', curr_proteins_list
        #
        if curr_proteins_list:
            if curr_sentence_id not in sentence_id_proteins_list_map:
                sentence_id_proteins_list_map[curr_sentence_id] = curr_proteins_list
            else:
                for curr_protein in curr_proteins_list:
                    if curr_protein not in sentence_id_proteins_list_map[curr_sentence_id]:
                        sentence_id_proteins_list_map[curr_sentence_id].append(curr_protein)
    #
    return sentence_id_proteins_list_map


def filter_inferred_interactions_on_sentence_ids(inferred_interactions_list, labels_test_pred_prob, sentence_ids_list):
    new_inferred_interactions_list = []
    if labels_test_pred_prob is not None:
        new_labels_test_pred_prob = []
    else:
        new_labels_test_pred_prob = None
    #
    curr_idx = -1
    for curr_interaction in inferred_interactions_list:
        curr_idx += 1
        #
        if labels_test_pred_prob is not None:
            curr_prob = labels_test_pred_prob[curr_idx]
        else:
            curr_prob = None
        #
        curr_path = curr_interaction[1]
        #
        curr_sentence_id = saefcd.get_passage_sentence_id_frm_joint_graph_path(curr_path)
        curr_path = None
        curr_sentence_id = 'ID'+str(curr_sentence_id)
        #
        if curr_sentence_id in sentence_ids_list:
            new_inferred_interactions_list.append(curr_interaction)
            #
            if curr_prob is not None:
                new_labels_test_pred_prob.append(curr_prob)
    #
    return new_inferred_interactions_list, new_labels_test_pred_prob


def get_sentence_id_interaction_types_list_map_in_extracted_interactions_str_tuples(extracted_interactions_list):
    sentence_id_interaction_types_map = {}
    #
    for curr_interaction in extracted_interactions_list:
        curr_path = curr_interaction[1]
        #
        curr_sentence_id = saefcd.get_passage_sentence_id_frm_joint_graph_path(curr_path)
        curr_path = None
        curr_sentence_id = 'ID'+str(curr_sentence_id)
        #
        print 'curr_interaction', curr_interaction
        curr_interaction_type = curr_interaction[0][0]
        print 'curr_interaction_type', curr_interaction_type
        #
        if curr_sentence_id not in sentence_id_interaction_types_map:
            sentence_id_interaction_types_map[curr_sentence_id] = [curr_interaction_type]
        else:
            if curr_interaction_type not in sentence_id_interaction_types_map[curr_sentence_id]:
                sentence_id_interaction_types_map[curr_sentence_id].append(curr_interaction_type)
    #
    return sentence_id_interaction_types_map


def filter_sentence_id_interactions_list_map_wd_proteins_list(sentence_id_interactions_list_map, sentence_id_proteins_list_map, sentence_id_interaction_types_list):
    #
    new_sentence_id_interactions_list_map = {}
    #
    for curr_sentence_id in sentence_id_interactions_list_map:
        interactions_list = sentence_id_interactions_list_map[curr_sentence_id]
        #
        new_interactions_list = []
        #
        for curr_interaction in interactions_list:
            print 'curr_interaction', curr_interaction
            if curr_sentence_id in sentence_id_interaction_types_list:
                if not saefcd.match_protein_name_with_gold_list(curr_interaction[0], sentence_id_interaction_types_list[curr_sentence_id]):
                    continue
            #
            curr_proteins_list = curr_interaction[1:]
            print 'curr_proteins_list', curr_proteins_list
            #
            do_proteins_match_wd_gold = True
            #
            for curr_protein in curr_proteins_list:
                if curr_sentence_id in sentence_id_proteins_list_map:
                    if not saefcd.match_protein_name_with_gold_list(curr_protein, sentence_id_proteins_list_map[curr_sentence_id]):
                        do_proteins_match_wd_gold = False
                        break
            #
            if do_proteins_match_wd_gold:
                new_interactions_list.append(curr_interaction)
        #
        if new_interactions_list:
            new_sentence_id_interactions_list_map[curr_sentence_id] = new_interactions_list
    #
    return new_sentence_id_interactions_list_map


def filter_sentence_id_interactions_list_map_wd_sentence_ids_list(
        sentence_id_interactions_list_map, sentence_ids_list):
    #
    new_sentence_id_interactions_list_map = {}
    #
    for curr_sentence_id in sentence_id_interactions_list_map:
        if curr_sentence_id in sentence_ids_list:
            new_sentence_id_interactions_list_map[curr_sentence_id] = sentence_id_interactions_list_map[curr_sentence_id]
    #
    return new_sentence_id_interactions_list_map


def filter_out_symmetric_duplicates(extracted_interactions_list, labels_test_pred_prob):
    new_extracted_interactions_list = []
    #
    if labels_test_pred_prob is not None:
        new_labels_test_pred_prob = []
    else:
        new_labels_test_pred_prob = None
    #
    print 'No. of interactions before symmetric filtering', len(extracted_interactions_list)
    #
    curr_idx = -1
    #
    for curr_interaction in extracted_interactions_list:
        #
        curr_idx += 1
        #
        print 'curr_interaction', curr_interaction
        curr_interaction_type = curr_interaction[0][0]
        print 'curr_interaction_type', curr_interaction_type
        #
        is_filter_out = False
        if saefcd.match_protein_name_with_gold_list(curr_interaction_type, saefcd.symmetric_types_list):
            curr_proteins_list = curr_interaction[0][1:]
            assert len(curr_proteins_list) == 2
            if curr_proteins_list[0] == curr_proteins_list[1]:
                raise AssertionError
            elif curr_proteins_list[0] < curr_proteins_list[1]:
                is_filter_out = True
        #
        if not is_filter_out:
            new_extracted_interactions_list.append(curr_interaction)
            #
            if labels_test_pred_prob is not None:
                new_labels_test_pred_prob.append(labels_test_pred_prob[curr_idx])
    #
    print 'No. of interactions after symmetric filtering', len(new_extracted_interactions_list)
    assert len(new_extracted_interactions_list) > 0
    #
    return new_extracted_interactions_list, new_labels_test_pred_prob


def get_list_of_interaction_types_in_chicago_map(chicago_sentence_id__interactions_list_map):
    interaction_types = []
    for curr_sentence_id in chicago_sentence_id__interactions_list_map:
        for curr_interaction in chicago_sentence_id__interactions_list_map[curr_sentence_id]:
            curr_interaction_type = curr_interaction[0]
            if curr_interaction_type not in interaction_types:
                interaction_types.append(curr_interaction_type)
    #
    print 'len(interaction_types)', len(interaction_types)
    return interaction_types


if __name__ == '__main__':
    is_evaluate_inferred = False
    #
    gp = 'gp'
    svm = 'svm'
    corex = 'corex'
    #
    algo_options = [gp, svm, corex]
    algo = gp
    assert algo in algo_options
    #
    is_filter_chicago_interactions_on_proteins_in_extracted = True
    is_filter_on_positive_interactions_in_chicago = True
    #
    if is_evaluate_inferred:
        if algo == gp:
            labels_test_pred_prob = np.load('./chicago inference gp/labels_test_pred_prob.npz')['arr_0']
            print labels_test_pred_prob.shape
            #
            sel_idx = np.where(labels_test_pred_prob > 0.5)
        elif algo == corex:
            labels_test_pred = np.load('./chicago inference corex/labels_test_pred.npz')['arr_0']
            labels_test_pred_prob = None
            print 'labels_test_pred.shape', labels_test_pred.shape
            #
            sel_idx = np.where(labels_test_pred == 1)
            print sel_idx
            labels_test_pred = None
        elif algo == svm:
            labels_test_pred = np.load('./chicago inference svm/labels_test_pred.npz')['arr_0']
            labels_test_pred_prob = None
            print 'labels_test_pred.shape', labels_test_pred.shape
            #
            sel_idx = np.where(labels_test_pred == 1)
            print sel_idx
            labels_test_pred = None
        else:
            raise AssertionError
        #
        extracted_interactions_list = np.load('./chicago_interactions_str_list.npz')['arr_0']
        # extracted_interactions_list = np.load('./amr_graphs_chicago_filtered_str_tuples.npz')['arr_0']
        print 'extracted_interactions_list.shape', extracted_interactions_list.shape
        #original in chicago
        chicago_sentence_id__interactions_list_map = get_positive_sentence__interactions_list_map(is_filter=False)
        #
        if is_filter_chicago_interactions_on_proteins_in_extracted:
            sentence_id_proteins_list_map_in_extracted_interactions \
                = get_list_of_proteins_in_extracted_interactions_str_tuples(extracted_interactions_list.tolist())
            #
            sentence_id_interaction_types_map_in_extracted_interactions \
                = get_sentence_id_interaction_types_list_map_in_extracted_interactions_str_tuples(extracted_interactions_list.tolist())
            #
            chicago_sentence_id__interactions_list_map \
                = filter_sentence_id_interactions_list_map_wd_proteins_list(
                chicago_sentence_id__interactions_list_map,
                sentence_id_proteins_list_map_in_extracted_interactions,
                sentence_id_interaction_types_map_in_extracted_interactions)
        #
        inferred_interactions_list = extracted_interactions_list[sel_idx]
        if labels_test_pred_prob is not None:
            labels_test_pred_prob = labels_test_pred_prob[sel_idx]
            labels_test_pred_prob = labels_test_pred_prob.tolist()
        print 'inferred_interactions_list.shape', inferred_interactions_list.shape
        sel_idx = None
        #
        inferred_interactions_list = inferred_interactions_list.tolist()
        print 'len(inferred_interactions_list)', len(inferred_interactions_list)
        #
        if is_filter_on_positive_interactions_in_chicago:
            inferred_interactions_list, labels_test_pred_prob \
                = filter_inferred_interactions_on_sentence_ids(inferred_interactions_list, labels_test_pred_prob, chicago_sentence_id__interactions_list_map.keys())
        #
        inferred_interactions_list, labels_test_pred_prob = filter_out_symmetric_duplicates(inferred_interactions_list, labels_test_pred_prob)
        #
        with open('./inferred_interactions_list.json', 'w') as f_inferred_interactions_list:
            json.dump(inferred_interactions_list, f_inferred_interactions_list, indent=4)
        #
        with open('./chicago_sentence_id__interactions_list_map.json', 'w') as f_chicago_sentence_id__interactions_list_map:
            json.dump(chicago_sentence_id__interactions_list_map, f_chicago_sentence_id__interactions_list_map, indent=4)
        #
        evaluate_inferred_interactions_list(inferred_interactions_list, chicago_sentence_id__interactions_list_map)
        #
        merge_chicago_nd_inferred(inferred_interactions_list, labels_test_pred_prob, chicago_sentence_id__interactions_list_map)
        #
        chicago_interaction_types = get_list_of_interaction_types_in_chicago_map(chicago_sentence_id__interactions_list_map)
        print json.dumps(chicago_interaction_types, indent=4)
        #
    else:
        is_load_classifier = True
        is_load_data = True
        is_positive_labels_only_fr_train = True
        #
        if not is_load_classifier:
            if not is_load_data:
                K_train = cpgkmjtd.join_parallel_computed_kernel_matrices_sparse(1)
                sssm.save_sparse_csr(cap.absolute_path+'./K_train', K_train)
            else:
                K_train = sssm.load_sparse_csr(cap.absolute_path+'./K_train.npz')
            print 'K_train.shape', K_train.shape
            print 'K_train.nnz', K_train.nnz
        else:
            K_train = None
        #
        print 'getting the test train matrix'
        if not is_load_data:
            K_test = cpgkmjtd.join_parallel_computed_kernel_matrices_sparse(340)
            sssm.save_sparse_csr(cap.absolute_path+'./K_test', K_test)
        else:
            K_test = sssm.load_sparse_csr(cap.absolute_path+'./K_test.npz')
        print 'K_test.shape', K_test.shape
        print 'K_test.nnz', K_test.nnz
        #
        if (not is_load_classifier) or is_positive_labels_only_fr_train:
            if not is_load_data:
                _, labels_train \
                    = te.get_data_joint(is_train=True, load_sentence_frm_dot_if_required=False, is_word_vectors=False)
                np.savez_compressed(cap.absolute_path+'./labels_train', labels_train)
            else:
                labels_train = np.load(cap.absolute_path+'./labels_train.npz')['arr_0']
            #
            idx_label2 = np.where(labels_train == 2)
            labels_train[idx_label2] = 0
        else:
            labels_train = None
        #
        if is_positive_labels_only_fr_train:
            positive_label_train_idx = np.where(labels_train == 1)[0]
            print 'positive_label_train_idx', positive_label_train_idx
            #
            labels_train = labels_train[positive_label_train_idx]
            #
            if K_train is not None:
                K_train = K_train[positive_label_train_idx, :]
                K_train = K_train.tocsc()
                K_train = K_train[:, positive_label_train_idx]
                K_train = K_train.tocsr()
            #
            K_test = K_test.tocsc()
            K_test = K_test[:, positive_label_train_idx]
            K_test = K_test.tocsr()
        #
        if algo == svm:
            print 'SVM ...'
            labels_test_pred_prob, labels_test_pred = vcd.classify_wd_svm(K_train, labels_train, K_test, is_load_classifier=True)
        elif algo == gp:
            print 'Gaussian process ...'
            labels_test_pred_prob \
                = vcd.classify_wd_gaussian_process(K_train, labels_train, K_test, is_load_classifier=True)
            #
            labels_test_pred = np.zeros(labels_test_pred_prob.shape)
            labels_test_pred[np.where(labels_test_pred_prob > 0.5)] = 1
        elif algo == corex:
            print 'Corex ...'
            labels_test_pred = vcd.classify_wd_corex(K_test)
            labels_test_pred_prob = None
        else:
            raise AssertionError
        #
        if labels_test_pred_prob is not None:
            np.savez_compressed(cap.absolute_path+'./labels_test_pred_prob', labels_test_pred_prob)
        #
        assert labels_test_pred is not None
        np.savez_compressed(cap.absolute_path+'./labels_test_pred', labels_test_pred)

