import numpy as np
import eval_divergence_frm_kernel as edk
import json


def write_json_candidate_interactions_str():
    file_name = './amr_graphs_chicago_filtered'
    print 'loading ...'
    amr_graphs_test = np.load(file_name+'.npy')
    print 'loaded.'
    n = amr_graphs_test.shape[0]
    #
    interactions_str_list = []
    for curr_idx in range(n):
        curr_amr_graph = amr_graphs_test[curr_idx, 0]
        curr_interaction = curr_amr_graph['tuple']
        curr_interaction_str \
            = edk.get_triplet_str_tuple(
            curr_interaction,
            is_concept_mapped_to_interaction=False)
        #
        curr_path = curr_amr_graph['path']
        curr_amr_graph = None
        #
        curr_tuple = [list(curr_interaction_str), curr_path]
        #
        interactions_str_list.append(curr_tuple)
    #
    interactions_str_list = np.array(interactions_str_list)
    #
    np.savez_compressed(file_name+'_str_tuples', interactions_str_list)


if __name__ == '__main__':
    write_json_candidate_interactions_str()

