import numpy as np
import validate_on_chicago_data as vcd
import train_extractor as te
import eval_divergence_frm_kernel as edk
import json


def get_model_data_idx(amr_graphs_org):
    n = amr_graphs_org.shape[0]
    assert amr_graphs_org.shape[1] == 1
    amr_graphs = []
    model_idx_list = []
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs_org[curr_idx, 0]
        #
        curr_path = curr_amr_graph_map['path']
        # print 'curr_path', curr_path
        #
        if 'json_darpa_model_amrs' in curr_path:
            model_idx_list.append(curr_idx)
    #
    model_idx_list = np.array(model_idx_list)
    return model_idx_list


if __name__ == '__main__':
    amr_graphs, labels = te.get_processed_train_joint_data()
    print 'setting label 2 to value 0'
    idx_label2 = np.where(labels == 2)
    labels[idx_label2] = 0
    print 'filtering out data containing only positive labels'
    positive_label_idx = np.where(labels == 1)[0]
    labels = None
    amr_graphs = amr_graphs[positive_label_idx, :]
    positive_label_idx = None
    print 'amr_graphs.shape', amr_graphs.shape
    #
    print 'filtering our chicago data ...'
    n = amr_graphs.shape[0]
    assert amr_graphs.shape[1] == 1
    chicago_idx = vcd.get_chicago_test_data_idx(amr_graphs)
    non_chicago_idx = np.setdiff1d(np.arange(0, n), chicago_idx)
    chicago_idx = None
    amr_graphs = amr_graphs[non_chicago_idx, :]
    non_chicago_idx = None
    print 'amr_graphs.shape', amr_graphs.shape
    n = amr_graphs.shape[0]
    assert amr_graphs.shape[1] == 1
    #
    print 'filtering out model data ...'
    n = amr_graphs.shape[0]
    assert amr_graphs.shape[1] == 1
    model_idx = get_model_data_idx(amr_graphs)
    non_model_idx = np.setdiff1d(np.arange(0, n), model_idx)
    model_idx = None
    amr_graphs = amr_graphs[non_model_idx, :]
    non_model_idx = None
    print 'amr_graphs.shape', amr_graphs.shape
    n = amr_graphs.shape[0]
    assert amr_graphs.shape[1] == 1
    #
    list_of_interaction_str_tuples = []
    for curr_idx in range(n):
        curr_amr_map = amr_graphs[curr_idx, 0]
        curr_tuple = curr_amr_map['tuple']
        curr_tuple_str = edk.get_triplet_str_tuple(curr_tuple)
        curr_tuple = None
        list_of_interaction_str_tuples.append(curr_tuple_str)
    #
    with open('./pub_med_45_interactions.json', 'w') as f:
        json.dump(list_of_interaction_str_tuples, f, indent=4)

