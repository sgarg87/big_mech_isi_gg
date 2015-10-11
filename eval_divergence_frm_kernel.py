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
import pickle
import validate as v
import difflib as dl
import extract_from_amr_dot as ead
import amr_sets
import multiprocessing as mp
from config_parallel_processing import *
from config_console_output import *
import parallel_computing as pc
import constants_absolute_path as cap
import json
import amr_sets_aimed


is_filter_interactions_with_invalid_proteins = True
#
is_compute_mmd = False
is_compute_mig = False
is_compute_kld_kd = False
is_compute_kld_knn = False
is_compute_dist_kernel = False
#
is_validate_mmd = False
is_validate_mig = False
is_validate_kld_kd = False
is_validate_kld_knn = False
is_validate_dist_kernel = True
#
is_validate_org_kernel = False
#
is_concept_mapped_to_interaction_default = False
#
is_random_validation = False
is_random_test = False
assert not(is_random_validation and is_random_test)
#
activate = 'activate'
signal = 'signal'
stimulate = 'stimulate'
#
inactivate = 'inactivate'
inhibit = 'inhibit'
impede = 'impede'
suppress = 'suppress'
diminish = 'diminish'
#
bind = 'bind'
complex = 'complex'
associate = 'associate'
dissociate = 'dissociate'
dimerize = 'dimerize'
homodimerize = 'homodimerize'
homodimer = 'homodimer'
heterodimerize = 'heterodimerize'
heterodimer = 'heterodimer'
form = 'form'
#
decrease = 'decrease'
degrade = 'degrade'
#
increase = 'increase'
express = 'express'
transcribe = 'transcribe'
produce = 'produce'
potentiate = 'potentiate'
enhance = 'enhance'
#
translocate = 'translocate'
localize = 'localize'
relocalize = 'relocalize'
recruit = 'recruit'
#
phosphorylate = 'phosphorylate'
hyperphosphorylate = 'hyperphosphorylate'


concepts_interaction_map = {
    activate: activate,
    signal: activate,
    stimulate: activate,
    inactivate: inactivate,
    inhibit: inactivate,
    impede: inactivate,
    suppress: inactivate,
    diminish: inactivate,
    bind: bind,
    complex: bind,
    associate: bind,
    dimerize: bind,
    homodimerize: bind,
    homodimer: bind,
    heterodimerize: bind,
    heterodimer: bind,
    form: bind,
    dissociate: bind,
    decrease: decrease,
    degrade: decrease,
    increase: increase,
    express: increase,
    transcribe: increase,
    produce: increase,
    potentiate: increase,
    enhance: increase,
    translocate: translocate,
    localize: translocate,
    relocalize: translocate,
    recruit: translocate,
    phosphorylate: phosphorylate,
    hyperphosphorylate: phosphorylate
}


div_tol = 1e-2

nn_list = [2, 3, 4, 5]

is_parallel = True

is_alternative_data = False


def get_triplet_str_tuple(curr_triplet_nodes_tuple, is_concept_mapped_to_interaction=is_concept_mapped_to_interaction_default):
    curr_triplet_str_tuple = []
    m = len(curr_triplet_nodes_tuple)
    for curr_idx in range(m):
        if curr_triplet_nodes_tuple[curr_idx] is not None:
            curr_str = curr_triplet_nodes_tuple[curr_idx].get_name_formatted()
            curr_triplet_str_tuple.append(curr_str)
        else:
            curr_triplet_str_tuple.append(None)
    if is_concept_mapped_to_interaction:
        if curr_triplet_str_tuple[0] in concepts_interaction_map:
            curr_triplet_str_tuple[0] = concepts_interaction_map[curr_triplet_str_tuple[0]]
        else:
            print 'could not find interaction mapping for concept {}'.format(curr_triplet_str_tuple[0])
    return tuple(curr_triplet_str_tuple)


def find_matching_interaction(interactions_list, curr_str):
    # print 'curr_str', curr_str
    if curr_str in interactions_list:
        return curr_str
    elif curr_str.lower() in interactions_list:
        return curr_str.lower()
    elif curr_str.upper() in interactions_list:
        return curr_str.upper()
    #
    max_ratio = None
    max_ratio_interaction = None
    for curr_interaction_term in interactions_list:
        dl_obj = dl.SequenceMatcher(None, curr_interaction_term, curr_str)
        #
        curr_ratio = dl_obj.quick_ratio()
        if curr_ratio < 0.6:
            continue
        #
        is_vec_similar, cs = ead.is_word_similar(curr_str, interactions_list, cosine_cut_off=0.75)
        if not is_vec_similar:
            continue
        #
        max_ratio = max(max_ratio, curr_ratio)
        if max_ratio == curr_ratio:
            max_ratio_interaction = curr_interaction_term
    return max_ratio_interaction


def sort_amr_graphs_data_by_file_name(amr_graphs_org, labels_org, K_org):
    # amr_graphs_org also consist of dependencies
    n = amr_graphs_org.shape[0]
    assert amr_graphs_org.shape[1] == 1
    assert labels_org.size == n
    map_key_file_name__value_amr_dependency_map = {}
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs_org[curr_idx, 0]
        curr_label = labels_org[curr_idx]
        #
        curr_path = curr_amr_graph_map['path']
        curr_path_split = os.path.split(curr_path)
        curr_path_filename = curr_path_split[-1]
        print '+++++++++++++++++++++++++++++++++++++++'
        print 'curr_path', curr_path
        print 'curr_path_filename', curr_path_filename
        curr_path = None
        if curr_path_filename not in map_key_file_name__value_amr_dependency_map:
            map_key_file_name__value_amr_dependency_map[curr_path_filename] = []
        map_key_file_name__value_amr_dependency_map[curr_path_filename].append((curr_amr_graph_map, curr_label, curr_idx))
        if len(map_key_file_name__value_amr_dependency_map[curr_path_filename]) > 2:
            print 'curr_path_filename', curr_path_filename
            raise AssertionError
    #
    amr_graphs_org = None
    labels_org = None
    #
    print 'map_key_file_name__value_amr_dependency_map.keys()', map_key_file_name__value_amr_dependency_map.keys()
    #
    amr_graphs = []
    labels = []
    idx_list = []
    for curr_amr_dep_list in map_key_file_name__value_amr_dependency_map.values():
        assert len(curr_amr_dep_list) <= 2
        for curr_graph_map, curr_label, curr_idx in curr_amr_dep_list:
            amr_graphs.append(curr_graph_map)
            labels.append(curr_label)
            idx_list.append(curr_idx)
    assert len(amr_graphs) == n
    amr_graphs = np.array(amr_graphs, dtype=np.object).reshape((n, 1))
    labels = np.array(labels)
    assert len(labels) == n
    idx_list = np.array(idx_list)
    assert len(idx_list) == n
    K = K_org[np.meshgrid(idx_list.astype(np.int32), idx_list.astype(np.int32), indexing='ij', sparse=True, copy=False)]
    assert K.shape[0] == n and K.shape[1] == n
    return amr_graphs, labels, K


def get_amr_set_fr_path(curr_path):
    print 'curr_path', curr_path
    sel_amr_set = None
    if is_alternative_data:
        amr_sets_list = amr_sets_aimed.amr_sets_list_merged
    else:
        amr_sets_list = amr_sets.amr_sets_list_all
    for curr_amr_set in amr_sets_list:
        if curr_amr_set in curr_path:
            if sel_amr_set is not None:
                if sel_amr_set == curr_amr_set:
                    raise AssertionError
                elif len(sel_amr_set) < len(curr_amr_set):
                    assert sel_amr_set in curr_amr_set
                    sel_amr_set = curr_amr_set
                elif len(sel_amr_set) > len(curr_amr_set):
                    assert curr_amr_set in sel_amr_set
                else:
                    raise AssertionError
            else:
                sel_amr_set = curr_amr_set
    assert sel_amr_set is not None, curr_path
    print '({},{})'.format(curr_path, sel_amr_set)
    return sel_amr_set


def match_protein_name(protein, protein_gold, min_ratio=0.95):
    is_match = False
    if protein == protein_gold:
        is_match = True
    elif protein.lower() == protein_gold.lower():
        is_match = True
    # else:
    #     dl_obj = dl.SequenceMatcher(None, protein, protein_gold)
    #     curr_ratio = dl_obj.quick_ratio()
    #     if curr_ratio > min_ratio:
    #         is_match = True
    #
    return is_match


def match_protein_name_with_gold_list(protein, proteins_gold_list):
    is_match = False
    for curr_gold_protein in proteins_gold_list:
        if match_protein_name(protein, curr_gold_protein):
            is_match = True
            break
    return is_match


def load_valid_proteins():
    with open(cap.absolute_path+'./valid_proteins_in_train_data.json', 'r') as f:
        valid_proteins_list = json.load(f)
    assert valid_proteins_list is not None and valid_proteins_list
    return valid_proteins_list


def filter_amr_graphs_data_invalid_proteins(interaction_idx_list_map):
    valid_proteins_list = load_valid_proteins()
    #
    selected_idx = []
    interactions_list = interaction_idx_list_map.keys()
    n = len(interactions_list)
    #
    for i in range(n):
        is_invalid = False
        #
        curr_label = interactions_list[i][1]
        if curr_label not in [1, 2]:
            curr_interaction_str_tuple = interactions_list[i][0]
            #
            list_of_proteins = list(curr_interaction_str_tuple[1:])
            assert len(list_of_proteins) in [2, 3]
            #
            if None in list_of_proteins:
                list_of_proteins.remove(None)
            #
            for curr_protein in list_of_proteins:
                if not isinstance(curr_protein, unicode):
                    curr_protein = unicode(curr_protein, 'utf-8')
                if not match_protein_name_with_gold_list(curr_protein, valid_proteins_list):
                    print '*************************'
                    print 'curr_protein', curr_protein
                    is_invalid = True
                    break
        #
        if not is_invalid:
            selected_idx.append(i)
    #
    print 'len(selected_idx)', len(selected_idx)
    return selected_idx


def get_statistics_amr_graphs_fr_interaction(amr_graphs, labels):
    print 'amr_graphs.shape', amr_graphs.shape
    n = amr_graphs.size
    interaction_count_map = {}
    interaction_idx_list_map = {}
    interaction_paths_list_map = {}
    path_file_name_keys_list__map = {}
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs[curr_idx, 0]
        if debug:
            print curr_amr_graph_map
        curr_triplet_nodes_tuple = curr_amr_graph_map['tuple']
        curr_path = curr_amr_graph_map['path']
        #
        curr_path_split = os.path.split(curr_path)
        curr_path_filename = curr_path_split[-1]
        #
        if curr_path_filename not in path_file_name_keys_list__map:
            path_file_name_keys_list__map[curr_path_filename] = {}
        #
        curr_triplet_str_tuple = get_triplet_str_tuple(curr_triplet_nodes_tuple)
        #
        if is_alternative_data:
            curr_triplet_str_tuple = list(curr_triplet_str_tuple[1:])
            if None in curr_triplet_str_tuple:
                curr_triplet_str_tuple.remove(None)
            curr_triplet_str_tuple.sort()
            curr_triplet_str_tuple = tuple(curr_triplet_str_tuple)
        #
        curr_amr_set = get_amr_set_fr_path(curr_path)
        curr_key = (curr_triplet_str_tuple, labels[curr_idx], curr_amr_set)
        # print 'curr_key', curr_key
        #
        is_process_sdg_too = False
        # note that value stores in keys 'sdg' and 'amr' are of different kind
        if '_sdg' in curr_path:
            # if amr is already encountered, then this sdg should be added for key of amr instead
            if 'amr' in path_file_name_keys_list__map[curr_path_filename]:
                curr_key = path_file_name_keys_list__map[curr_path_filename]['amr']
            else:
                # sdg came before amr for same relation
                # so, sdg to be processed later when amr is encountered for same relation
                path_file_name_keys_list__map[curr_path_filename]['sdg'] = (curr_idx, curr_path)
                continue
        else:
            path_file_name_keys_list__map[curr_path_filename]['amr'] = curr_key
            # sdg already came before this amr
            # so process sdg now (since its processing was skipped previously)
            if 'sdg' in path_file_name_keys_list__map[curr_path_filename]:
                is_process_sdg_too = True
                curr_idx_sdg, curr_path_sdg = path_file_name_keys_list__map[curr_path_filename]['sdg']
        #
        if curr_key not in interaction_count_map:
            interaction_count_map[curr_key] = 0
        #
        if curr_key not in interaction_idx_list_map:
            interaction_idx_list_map[curr_key] = []
        #
        if curr_key not in interaction_paths_list_map:
            interaction_paths_list_map[curr_key] = []
        interaction_count_map[curr_key] += 1
        interaction_idx_list_map[curr_key].append(curr_idx)
        interaction_paths_list_map[curr_key].append(curr_path)
        #
        if is_process_sdg_too:
            # sdg is added to same key
            interaction_count_map[curr_key] += 1
            interaction_idx_list_map[curr_key].append(curr_idx_sdg)
            interaction_paths_list_map[curr_key].append(curr_path_sdg)
    #
    if debug:
        print interaction_count_map
        print interaction_count_map.values()
    #
    if not ch.is_hpcc:
        plt.plot(interaction_count_map.values(), 'kx')
        plt.savefig(cap.absolute_path+'./interaction_num_graphs_plot.pdf', dpi=300, format='pdf')
    #
    with open(cap.absolute_path+'./interaction_num_graphs_stats.csv', 'w') as f:
        col_interaction_tuple = 'interaction tuple'
        col_count_of_graphs = 'count of graphs'
        col_label = 'label'
        col_amr_set = 'amr_set'
        field_names = [col_interaction_tuple, col_label, col_amr_set, col_count_of_graphs]
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for curr_interaction_tuple in interaction_count_map:
            writer.writerow(
                {
                    col_interaction_tuple: curr_interaction_tuple[0],
                    col_label: curr_interaction_tuple[1],
                    col_amr_set: curr_interaction_tuple[2],
                    col_count_of_graphs: interaction_count_map[curr_interaction_tuple]
                }
            )
    #
    return interaction_count_map, interaction_idx_list_map, interaction_paths_list_map


def get_statistics_dependency_graphs_fr_interaction(sdg_graphs, labels):
    print sdg_graphs.shape
    n = sdg_graphs.size
    interaction_count_map = {}
    interaction_idx_list_map = {}
    interaction_paths_list_map = {}
    path_file_name_keys_list__map = {}
    for curr_idx in range(n):
        curr_sdg_graph_map = sdg_graphs[curr_idx, 0]
        if debug:
            print curr_sdg_graph_map
        curr_triplet_nodes_tuple = curr_sdg_graph_map['tuple']
        curr_path = curr_sdg_graph_map['path']
        #
        curr_triplet_str_tuple = get_triplet_str_tuple(curr_triplet_nodes_tuple)
        #
        if is_alternative_data:
            curr_triplet_str_tuple = list(curr_triplet_str_tuple[1:])
            if None in curr_triplet_str_tuple:
                curr_triplet_str_tuple.remove(None)
            curr_triplet_str_tuple.sort()
            curr_triplet_str_tuple = tuple(curr_triplet_str_tuple)
        #
        curr_amr_set = get_amr_set_fr_path(curr_path)
        curr_key = (curr_triplet_str_tuple, labels[curr_idx], curr_amr_set)
        #
        if curr_key not in interaction_count_map:
            interaction_count_map[curr_key] = 0
        #
        if curr_key not in interaction_idx_list_map:
            interaction_idx_list_map[curr_key] = []
        #
        if curr_key not in interaction_paths_list_map:
            interaction_paths_list_map[curr_key] = []
        interaction_count_map[curr_key] += 1
        interaction_idx_list_map[curr_key].append(curr_idx)
        interaction_paths_list_map[curr_key].append(curr_path)
    #
    if debug:
        print interaction_count_map
        print interaction_count_map.values()
    #
    if not ch.is_hpcc:
        plt.plot(interaction_count_map.values(), 'kx')
        plt.savefig(cap.absolute_path+'./interaction_num_dep_graphs_plot.pdf', dpi=300, format='pdf')
    #
    return interaction_count_map, interaction_idx_list_map, interaction_paths_list_map


def filter_amr_graphs_data(is_amr_else_dep, amr_graphs_org, labels_org, K_org):
    n = amr_graphs_org.shape[0]
    assert amr_graphs_org.shape[1] == 1
    assert labels_org.size == n
    amr_graphs = []
    labels = []
    idx_list = []
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs_org[curr_idx, 0]
        curr_label = labels_org[curr_idx]
        #
        curr_path = curr_amr_graph_map['path']
        #
        if ('_sdg' in curr_path and (not is_amr_else_dep)) or ('_sdg' not in curr_path and is_amr_else_dep):
            amr_graphs.append(curr_amr_graph_map)
            labels.append(curr_label)
            idx_list.append(curr_idx)
    #
    n = len(amr_graphs)
    amr_graphs = np.array(amr_graphs, dtype=np.object).reshape((n, 1))
    labels = np.array(labels)
    assert len(labels) == n
    assert len(idx_list) == n
    idx_list = np.array(idx_list)
    assert len(idx_list) == n
    K = K_org[np.meshgrid(idx_list.astype(np.int32), idx_list.astype(np.int32), indexing='ij', sparse=True, copy=False)]
    assert K.shape[0] == n and K.shape[1] == n
    print 'after filtering ...'
    print K.shape
    print amr_graphs.shape
    print labels.shape
    return amr_graphs, labels, K


def set_amr_sdg_kernel_zero(amr_graphs_org, K_org):
    #amr graphs org can contain both amr and sdg
    n = amr_graphs_org.shape[0]
    assert amr_graphs_org.shape[1] == 1
    sdg_idx_list = []
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs_org[curr_idx, 0]
        #
        curr_path = curr_amr_graph_map['path']
        #
        if '_sdg' in curr_path:
            sdg_idx_list.append(curr_idx)
    sdg_idx_list = np.array(sdg_idx_list)
    #
    others_idx_list = np.setdiff1d(np.arange(n), sdg_idx_list)
    assert (sdg_idx_list.size + others_idx_list.size) == n
    assert np.intersect1d(sdg_idx_list, others_idx_list).size == 0
    #
    K[np.meshgrid(sdg_idx_list.astype(np.int32), others_idx_list.astype(np.int32), indexing='ij', sparse=True)] = 0
    K[np.meshgrid(others_idx_list.astype(np.int32), sdg_idx_list.astype(np.int32), indexing='ij', sparse=True)] = 0
    assert np.all(K[np.meshgrid(sdg_idx_list.astype(np.int32), others_idx_list.astype(np.int32), indexing='ij', sparse=True, copy=False)] == 0)
    assert np.all(K[np.meshgrid(others_idx_list.astype(np.int32), sdg_idx_list.astype(np.int32), indexing='ij', sparse=True, copy=False)] == 0)
    assert not np.all(K[np.meshgrid(sdg_idx_list.astype(np.int32), sdg_idx_list.astype(np.int32), indexing='ij', sparse=True, copy=False)] == 0)
    assert not np.all(K[np.meshgrid(others_idx_list.astype(np.int32), others_idx_list.astype(np.int32), indexing='ij', sparse=True, copy=False)] == 0)
    assert not np.all(K == 0)
    return K


def compute_divergence_wrapper(
        K,
        interaction_idx_list_map,
        interaction_keys_list1,
        interaction_keys_list2,
        # input_data_queue,
        divergence_algo,
        divergence_matrix_queue
    ):
    try:
        # K = input_data_queue.get()
        assert K is not None
        assert np.all(K >= 0)
        assert np.all(K <= 1)
        #
        # interaction_idx_list_map = input_data_queue.get()
        assert interaction_idx_list_map is not None and interaction_idx_list_map
        #
        # interaction_keys_list1 = input_data_queue.get()
        assert interaction_keys_list1 is not None and interaction_keys_list1
        #
        # interaction_keys_list2 = input_data_queue.get()
        assert interaction_keys_list2 is not None and interaction_keys_list2
        #
        D = compute_divergence(K, interaction_idx_list_map, interaction_keys_list1, interaction_keys_list2, divergence_algo)
        assert D is not None
        assert_divergence(D, divergence_algo)
        #
        divergence_matrix_queue.put(D)
    except BaseException as e:
        print 'error in the subprocess (base exception)'
        print e
        divergence_matrix_queue.put(e)
    except OSError as ee:
        print 'error in the subprocess (os error)'
        print ee
        divergence_matrix_queue.put(ee)


def initialize_divergence(divergence_algo, n1, n2):
    if divergence_algo != 'kl_knn':
        D = -1*np.ones(shape=(n1, n2))
    else:
        D = {}
        for curr_nn in nn_list:
            D[curr_nn] = -1*np.ones(shape=(n1, n2))
    return D


def compute_divergence(K, interaction_idx_list_map, interaction_keys_list1, interaction_keys_list2, divergence_algo):
    #
    assert divergence_algo in ['mmd', 'mi_gaussian', 'kl_kd', 'kl_knn', 'k']
    #
    assert np.all(K >= 0) and np.all(K <= 1)
    #
    assert interaction_idx_list_map is not None and interaction_idx_list_map
    #
    assert interaction_keys_list1 is not None and interaction_keys_list1
    #
    assert interaction_keys_list2 is not None and interaction_keys_list2
    #
    n1 = len(interaction_keys_list1)
    n2 = len(interaction_keys_list2)
    #
    D = initialize_divergence(divergence_algo, n1, n2)
    #
    start_time = time.time()
    print 'Computing divergence matrices ... \n'
    for i in range(n1):
        sys.stdout.write('.')
        sys.stdout.flush()
        i_idx_list = np.array(interaction_idx_list_map[interaction_keys_list1[i]])
        Kii = K[np.meshgrid(i_idx_list, i_idx_list, indexing='ij', sparse=True, copy=False)]
        #todo: only n(n-1)/2 computations required, make the change
        if is_parallel:
            j_range = range(n2)
        else:
            j_range = range(i+1)
        for j in j_range:
            j_idx_list = np.array(interaction_idx_list_map[interaction_keys_list2[j]])
            Kjj = K[np.meshgrid(j_idx_list.astype(np.int32), j_idx_list.astype(np.int32), indexing='ij', sparse=True, copy=False)]
            Kij = K[np.meshgrid(i_idx_list.astype(np.int32), j_idx_list.astype(np.int32), indexing='ij', sparse=True, copy=False)]
            #
            if divergence_algo == 'mmd':
                D[i, j] = eval_divergence(Kii, Kjj, Kij, algo=divergence_algo)
                if not is_parallel:
                    D[j,i] = D[i,j]
            elif divergence_algo == 'k':
                D[i, j] = eval_divergence(Kii, Kjj, Kij, algo=divergence_algo)
                if not is_parallel:
                    D[j,i] = D[i,j]
            elif divergence_algo == 'mi_gaussian':
                D[i, j] = eval_divergence(Kii, Kjj, Kij, algo=divergence_algo)
                if not is_parallel:
                    D[j,i] = D[i,j]
            elif divergence_algo == 'kl_kd':
                D[i, j] = eval_divergence(Kii, Kjj, Kij, algo=divergence_algo)
                if not is_parallel:
                    D[j,i] = D[i,j]
            elif divergence_algo == 'kl_knn':
                for curr_nn in nn_list:
                    D[curr_nn][i, j] = eval_divergence(Kii, Kjj, Kij, algo=divergence_algo, nn=curr_nn)
                    if not is_parallel:
                        D[curr_nn][j,i] = D[curr_nn][i,j]
            else:
                raise AssertionError
    print 'Time to compute the divergence matrices was {}'.format(time.time()-start_time)
    assert D is not None
    assert_divergence(D, divergence_algo)
    return D


def assert_divergence(D, divergence_algo):
    assert D is not None
    if divergence_algo == 'kl_knn':
        for curr_nn in nn_list:
            assert np.all(D[curr_nn] >= 0) and np.all(D[curr_nn] <= 1)
    else:
        assert np.all(D >= 0)


def compute_divergence_parallel(K, interaction_idx_list_map, interaction_keys_list1, interaction_keys_list2, divergence_algo):
    divergence_matrix_queue = [mp.Queue() for d in range(num_cores)]
    #
    n1 = len(interaction_keys_list1)
    n2 = len(interaction_keys_list2)
    #
    D = initialize_divergence(divergence_algo, n1, n2)
    #
    idx_range_parallel = pc.uniform_distribute_tasks_across_cores(n1, num_cores)
    #
    # input_data_queue = [mp.Queue() for d in range(num_cores)]
    # for currCore in range(num_cores):
    #     curr_core_interaction_keys_list1 = [interaction_keys_list1[curr_idx_in_core] for curr_idx_in_core in idx_range_parallel[currCore]]
    #     input_data_queue[currCore].put(K)
    #     input_data_queue[currCore].put(interaction_idx_list_map)
    #     input_data_queue[currCore].put(curr_core_interaction_keys_list1)
    #     input_data_queue[currCore].put(interaction_keys_list2)
    #
    processes = [
        mp.Process(
            target=compute_divergence_wrapper,
            args=(
                K,
                interaction_idx_list_map,
                [interaction_keys_list1[curr_idx_in_core] for curr_idx_in_core in idx_range_parallel[currCore]],
                interaction_keys_list2,
                divergence_algo,
                # input_data_queue[currCore],
                divergence_matrix_queue[currCore]
            )
        ) for currCore in range(num_cores)
    ]
    #start processes
    for process in processes:
        process.start()
    for currCore in range(num_cores):
        print('waiting for results from core ', currCore)
        # todo: this should not work, replace with mesh index
        result = divergence_matrix_queue[currCore].get()
        assert result is not None
        if isinstance(result, BaseException) or isinstance(result, OSError): #it means that subprocess has an error
            print 'a child processed has thrown an exception. raising the exception in the parent process to terminate the program'
            print 'one of the child processes failed, so killing all child processes'
            #kill all subprocesses
            for process in processes:
                if process.is_alive():
                    process.terminate() #assuming that the child process do not have its own children (those granchildren would be orphaned with terminate() if any)
            print 'killed all child processes'
            raise result
        else:
            assert_divergence(result, divergence_algo)
            if divergence_algo != 'kl_knn':
                D[idx_range_parallel[currCore], :] = result
                assert D[idx_range_parallel[currCore], :].shape == result.shape, str(result.shape)
            else:
                for curr_nn in nn_list:
                    D[curr_nn][idx_range_parallel[currCore], :] = result[curr_nn]
                    assert D[curr_nn][idx_range_parallel[currCore], :].shape == result[curr_nn].shape, str(result[curr_nn].shape)
        print('got results from core ', currCore)
    kernel_matrix_queue = None
    #wait for processes to complete
    for process in processes:
        process.join()
    assert_divergence(D, divergence_algo)
    return D


def eval_divergence_matrix(amr_graphs, labels, K, is_amr_only=False, is_dep_only=False, has_data_only_amrs=False):
    #
    assert K.shape[0] == K.shape[1] and len(K.shape) == 2
    assert amr_graphs.shape[0] == K.shape[0]
    assert labels.size == amr_graphs.shape[0]
    #
    assert not (is_amr_only and is_dep_only)
    #
    if not has_data_only_amrs:
        if (not is_amr_only) and (not is_dep_only):
            # todo: this sorting can be avoided if we are selecting test set based on random selection or paper selection
            # todo: whereas it is required if we do continuous subset selection from dataset (later is not recommended
            # todo: since its is difficult to ensure that the data before continuous subselection is not randomize
            # todo: in some method because of some unexpected reason (it happended in the past when we pust all amrs
            # todo: for same interaction in one set))
            amr_graphs, labels, K = sort_amr_graphs_data_by_file_name(amr_graphs, labels, K)
        elif is_amr_only:
            amr_graphs, labels, K \
                = filter_amr_graphs_data(is_amr_else_dep=True, amr_graphs_org=amr_graphs, labels_org=labels, K_org=K)
        elif is_dep_only:
            amr_graphs, labels, K \
                = filter_amr_graphs_data(is_amr_else_dep=False, amr_graphs_org=amr_graphs, labels_org=labels, K_org=K)
    #
    # todo: get statistics must come after sort and filter since it returns final indices list for each interaction
    #
    if is_dep_only:
        assert not has_data_only_amrs
        interaction_count_map, interaction_idx_list_map, interaction_paths_list_map \
            = get_statistics_dependency_graphs_fr_interaction(amr_graphs, labels)
    else:
        interaction_count_map, interaction_idx_list_map, interaction_paths_list_map \
            = get_statistics_amr_graphs_fr_interaction(amr_graphs, labels)
    # take care of efficiency later on
    n = len(interaction_idx_list_map)
    if coarse_debug:
        print 'No. of sample sets are {}'.format(n)
    #
    interaction_keys_list = interaction_idx_list_map.keys()
    #
    labels_fr_sets = []
    for i in range(n):
        labels_fr_sets.append(interaction_keys_list[i][1])
    labels_fr_sets = np.array(labels_fr_sets)
    #
    prefix = 'divergence_matrices/'
    suffix = str(n)+'_'+str(n)
    if is_compute_mmd or is_compute_mig or is_compute_kld_kd or is_compute_kld_knn or is_compute_dist_kernel:
        if is_compute_mmd:
            if is_parallel:
                Dmmd = compute_divergence_parallel(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='mmd')
            else:
                Dmmd = compute_divergence(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='mmd')
            assert_divergence(Dmmd, 'mmd')
            np.save(cap.absolute_path+prefix+'maximum_mean_discrepancy'+suffix, Dmmd)
        if is_compute_dist_kernel:
            if is_parallel:
                Dk = compute_divergence_parallel(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='k')
            else:
                Dk = compute_divergence(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='k')
            assert_divergence(Dk, 'k')
            np.save(cap.absolute_path+prefix+'distribution_kernel'+suffix, Dk)
        if is_compute_mig:
            if is_parallel:
                Dmig = compute_divergence_parallel(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='mi_gaussian')
            else:
                Dmig = compute_divergence(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='mi_gaussian')
            assert_divergence(Dmig, 'mi_gaussian')
            np.save(cap.absolute_path+prefix+'mutual_information_gaussian_process'+suffix, Dmig)
        if is_compute_kld_kd:
            if is_parallel:
                Dkld_kd = compute_divergence_parallel(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='kl_kd')
            else:
                Dkld_kd = compute_divergence(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='kl_kd')
            assert_divergence(Dkld_kd, 'kl_kd')
            np.save(cap.absolute_path+prefix+'kl-divergence_kernel_density'+suffix, Dkld_kd)
        if is_compute_kld_knn:
            if is_parallel:
                Dkld_knn = compute_divergence_parallel(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='kl_knn')
            else:
                Dkld_knn = compute_divergence(K, interaction_idx_list_map, interaction_keys_list, interaction_keys_list, divergence_algo='kl_knn')
            assert_divergence(Dkld_knn, 'kl_knn')
            for curr_nn in nn_list:
                np.save(cap.absolute_path+prefix+'kl-divergence_{}nn'.format(curr_nn)+suffix, Dkld_knn[curr_nn])
    #
    if not (is_compute_mmd and is_compute_mig and is_compute_kld_kd and is_compute_kld_knn and is_compute_dist_kernel):
        suffix += '.npy'
        if is_validate_mmd and (not is_compute_mmd):
            Dmmd = np.load(cap.absolute_path+prefix+'maximum_mean_discrepancy'+suffix)
        if is_validate_dist_kernel and (not is_compute_dist_kernel):
            Dk = np.load(cap.absolute_path+prefix+'distribution_kernel'+suffix)
        if is_validate_mig and (not is_compute_mig):
            Dmig = np.load(cap.absolute_path+prefix+'mutual_information_gaussian_process'+suffix)
        if is_validate_kld_kd and (not is_compute_kld_kd):
            Dkld_kd = np.load(cap.absolute_path+prefix+'kl-divergence_kernel_density'+suffix)
        if is_validate_kld_knn and (not is_compute_kld_knn):
            Dkld_knn = {}
            for curr_nn in nn_list:
                Dkld_knn[curr_nn] = np.load(cap.absolute_path+prefix+'kl-divergence_{}nn'.format(curr_nn)+suffix)
    #
    if is_filter_interactions_with_invalid_proteins:
        filtered_interaction_idx_list = filter_amr_graphs_data_invalid_proteins(interaction_idx_list_map)
    else:
        filtered_interaction_idx_list = None
    #
    vps_obj = v.ValidatePaperSets(K, amr_graphs, interaction_idx_list_map,
                                  filtered_interaction_idx_list=filtered_interaction_idx_list,
                                  is_alternative_data=is_alternative_data)
    if not is_random_validation and not is_random_test:
        vps_obj.sample_train_test_interaction_random_subset_at_paper_level()
    elif is_random_test:
        vps_obj.sample_train_test_random_subset()
    elif is_random_validation:
        vps_obj.sample_test_random_subset_rest_train()
    #
    if is_validate_mmd:
        print 'validating on maximum mean discrepancy ...'
        vps_obj.validate_at_paper_level(np.exp(-Dmmd), labels_fr_sets, 'maximum_mean_discrepancy', is_divergence=True)
    if is_validate_dist_kernel:
        print 'validating on distribution ...'
        vps_obj.validate_at_paper_level(Dk, labels_fr_sets, 'distribution_kernel', is_divergence=True)
    if is_validate_kld_kd:
        print 'validating on kl divergence (kernel density) ...'
        vps_obj.validate_at_paper_level(np.exp(-Dkld_kd), labels_fr_sets, 'kl_divergence__kernel_density', is_divergence=True)
    if is_validate_mig:
        print 'validating on mutual information Gaussian ...'
        vps_obj.validate_at_paper_level(Dmig, labels_fr_sets, 'mutual_information_gaussian_process', is_divergence=True)
    if is_validate_kld_knn:
        for curr_nn in nn_list:
            print 'validating on kl divergence {}-nn'.format(curr_nn)
            vps_obj.validate_at_paper_level(np.exp(-Dkld_knn[curr_nn]), labels_fr_sets, 'kl_divergence__{}nn'.format(curr_nn), is_divergence=True)
    if is_validate_org_kernel:
        print 'validating on original kernel (also with maximum likely inferred) ...'
        vps_obj.validate_at_paper_level(K, labels, 'original_kernel', is_divergence=False)


def eval_divergence(Kii, Kjj, Kij, algo, nn=None):
    #
    # nn is no. of nearest neighbors
    #
    # kl divergence estimation using kernel density
    # kl divergence estimation with kernel tricky in k-nn
    # kl divergence estimation with joint of kernel density and kernel trick in k-nn
    # mutual information estimation with each sample as Gaussian (Gaussian process)
    # maximum mean discrepancy method
    assert Kii.shape[0] == Kii.shape[1] and len(Kii.shape) == 2
    assert Kjj.shape[0] == Kjj.shape[1] and len(Kjj.shape) == 2
    assert len(Kij.shape) == 2 and Kii.shape[0] == Kij.shape[0] and Kjj.shape[0] == Kij.shape[1]
    # maximum mean discrepancy
    if algo == 'mmd':
        assert nn is None
        return eval_max_mean_discrepancy(Kii, Kjj, Kij)
    if algo == 'k':
        assert nn is None
        return eval_distribution_kernel(Kij)
    # mutual information with Gaussian on each sample
    elif algo == 'mi_gaussian':
        assert nn is None
        return eval_mutual_information_multivariate_gaussian(Kii, Kjj, Kij)
    # kl divergence with kernel density estimation
    elif algo == 'kl_kd':
        assert nn is None
        return eval_kl_div_kernel_density(Kii, Kjj, Kij)
    elif algo == 'kl_knn':
        assert nn is not None
        return eval_kl_div_knn(Kii, Kjj, Kij, nn)
    else:
        raise NotImplementedError


def eval_max_mean_discrepancy(Kii, Kjj, Kij):
    start_time = time.time()
    maximum_mean_discrepancy = Kii.mean() + Kjj.mean() - 2*Kij.mean()
    if maximum_mean_discrepancy < 0:
        if -div_tol < maximum_mean_discrepancy < 0:
            maximum_mean_discrepancy = 0
        else:
            print 'Kii.mean()', Kii.mean()
            print 'Kjj.mean()', Kjj.mean()
            print 'Kij.mean()', Kij.mean()
            print 'maximum_mean_discrepancy', maximum_mean_discrepancy
            raise AssertionError
    if coarse_debug:
        print 'time to compute maximum mean discrepancy was {}'.format(time.time()-start_time)
    return maximum_mean_discrepancy


def eval_distribution_kernel(Kij):
    return Kij.mean()


def eval_mutual_information_multivariate_gaussian(Kii, Kjj, Kij):
    start_time = time.time()
    if debug:
        print Kij
        print Kij.shape
    m1 = Kij.shape[0]
    if debug:
        print m1
    m2 = Kij.shape[1]
    if debug:
        print m2
    n = m1+m2
    if debug:
        print n
    K = -1*np.ones(shape=(n, n))
    K[np.meshgrid(range(m1), range(m1), indexing='ij', sparse=True)] = Kii
    K[np.meshgrid(range(m1), range(m1, n), indexing='ij', sparse=True)] = Kij
    K[np.meshgrid(range(m1, n), range(m1), indexing='ij', sparse=True)] = Kij.transpose()
    K[np.meshgrid(range(m1, n), range(m1, n), indexing='ij', sparse=True)] = Kjj
    const_epsilon = 1e-320
    det_K = np.linalg.det(K)
    if debug:
        print det_K
    if det_K <= 0:
        entropy_K = 0
    else:
        entropy_K = float(n)/2 + (float(n)*math.log(2*math.pi))/2 + math.log(det_K)
    if entropy_K < 0:
        entropy_K = 0
    #
    det_Kii = np.linalg.det(Kii)
    if debug:
        print det_Kii
    if det_Kii <= 0:
        entropy_Kii = 0
    else:
        entropy_Kii = float(m1)/2 + (float(m1)*math.log(2*math.pi))/2 + math.log(det_Kii)
    if entropy_Kii < 0:
        entropy_Kii = 0
    #
    det_Kjj = np.linalg.det(Kjj)
    if debug:
        print det_Kjj
    if det_Kjj <= 0:
        entropy_Kjj = 0
    else:
        entropy_Kjj = float(m2)/2 + (float(m2)*math.log(2*math.pi))/2 + math.log(det_Kjj)
    if entropy_Kjj < 0:
        entropy_Kjj = 0
    #
    mutual_information = entropy_Kii + entropy_Kjj - entropy_K
    if mutual_information < 0:
        if mutual_information < 0 and mutual_information > -div_tol:
            mutual_information = 0
        else:
            print 'det_K', det_K
            print 'det_Kii', det_Kii
            print 'det_Kjj', det_Kjj
            print 'entropy_K', entropy_K
            print 'entropy_Kii', entropy_Kii
            print 'entropy_Kjj', entropy_Kjj
            raise AssertionError
    #
    if coarse_debug:
        print 'time to compute mutual information with gaussian process was {}'.format(time.time()-start_time)
    return mutual_information


def eval_kl_div_kernel_density(Kii, Kjj, Kij):
    start_time = time.time()
    const_epsilon = 1e-30
    kl_ii_jj = np.log(np.divide(Kii.mean(1)+const_epsilon, Kij.mean(1)+const_epsilon)).mean()
    kl_jj_ii = np.log(np.divide(Kjj.mean(1)+const_epsilon, Kij.transpose().mean(1)+const_epsilon)).mean()
    kl = kl_ii_jj + kl_jj_ii
    if kl < 0:
        if -div_tol < kl < 0:
            kl = 0
        else:
            print 'Kii.mean(1)+const_epsilon', Kii.mean(1)+const_epsilon
            print 'Kjj.mean(1)+const_epsilon', Kjj.mean(1)+const_epsilon
            print 'Kij.mean(1)+const_epsilon', Kij.mean(1)+const_epsilon
            print 'Kij.transpose().mean(1)+const_epsilon', Kij.transpose().mean(1)+const_epsilon
            print 'kl_ii_jj', kl_ii_jj
            print 'kl_jj_ii', kl_jj_ii
            print 'kl', kl
            raise AssertionError
    if coarse_debug:
        print 'time to compute kl divergence with kernel density estimation was {}'.format(time.time()-start_time)
    return kl


def eval_kl_div_knn_frm_dist(Dii, Dij, no_of_neighbors):
    n = Dij.shape[0]
    m = Dij.shape[1]
    assert Dii.shape[0] == Dii.shape[1] == n
    #
    l = -1*np.ones(n)
    k = -1*np.ones(n)
    for i in range(n):
        curr_dist_in = np.copy(Dii[i, :])
        curr_dist_out = np.copy(Dij[i, :])
        nn_in = min(no_of_neighbors, curr_dist_in.size)
        nn_out = min(no_of_neighbors, curr_dist_out.size)
        dist_nn_in = np.partition(curr_dist_in, nn_in-1)[nn_in-1]
        dist_nn_out = np.partition(curr_dist_out, nn_out-1)[nn_out-1]
        dist_nn_max_of_in_out = max(dist_nn_in, dist_nn_out)
        if dist_nn_max_of_in_out == dist_nn_in:
            l[i] = nn_in
            k[i] = np.where(curr_dist_out <= dist_nn_max_of_in_out)[0].size
            assert k[i] >= nn_out
        elif dist_nn_max_of_in_out == dist_nn_out:
            k[i] = nn_out
            l[i] = np.where(curr_dist_in <= dist_nn_max_of_in_out)[0].size
            assert l[i] >= nn_in
        else:
            raise AssertionError
    kl_i_j = (ss.digamma(l)-ss.digamma(k)).mean() + math.log(m/float(n-1))
    if kl_i_j < 0:
        if -div_tol < kl_i_j < 0:
            kl_i_j = 0
        else:
            if debug:
                print 'kl_i_j', kl_i_j
                print 'l', l
                print 'k', k
                print 'ss.digamma(l)', ss.digamma(l)
                print 'ss.digamma(k)', ss.digamma(k)
                print '(ss.digamma(l)-ss.digamma(k)).mean()', (ss.digamma(l)-ss.digamma(k)).mean()
                print 'm', m
                print 'n', n
                print 'm/float(n-1)', m/float(n-1)
                print 'math.log(m/float(n-1))', math.log(m/float(n-1))
            # raise AssertionError
    return kl_i_j, l, k


def eval_kl_div_knn(Kii, Kjj, Kij, num_of_neighbors, algo_special_cond = 'kld_kd'):
    #
    n = Kij.shape[0]
    m = Kij.shape[1]
    assert Kii.shape[0] == Kii.shape[1] == n
    assert Kjj.shape[0] == Kjj.shape[1] == m
    #
    if m < num_of_neighbors or n < num_of_neighbors:
        # kernel divergence estimation with kernel density
        # for divergence between two points, this corresponds to -2log(k(i,j))
        # it seems to make more sense to use KL-D kernel density estimation,
        #  since the divergence between two points scale to [0, inf)
        if algo_special_cond == 'kld_kd':
            return eval_max_mean_discrepancy(Kii, Kjj, Kij)
        # maximum mean discrepancy
        # for divergence between two points, this corresponds to 2(1-K(i,j))
        # this basically represents the kernel trick distance. this also seems to make distance
        elif algo_special_cond == 'mmd':
            return eval_max_mean_discrepancy(Kii, Kjj, Kij)
    #applying kernel trick to evaluate distance matrices
    # in our case, the self similarity k(i,i) is always 1 since kernels are normalized
    # so expression for distance in terms of kernels simplifies as:
    # d(x_i, x_j) = k(x_i, xi) + k(x_j,x_j) - 2k(x_i,x_j)
    # d(x_i, x_j) = 1 + 1 - 2k(x_i,x_j)
    # d(x_i, x_j) = 2(1 - k(x_i,x_j))
    assert np.all(Kii.diagonal() == 1)
    assert np.all(Kjj.diagonal() == 1)
    Dii = 2*(1-Kii)
    Djj = 2*(1-Kjj)
    Dij = 2*(1-Kij)
    if debug:
        print 'num_of_neighbors', num_of_neighbors
    # # duplication doesn't work, so commented the code as of now
    # #
    # # add duplicate elements by replicating distances with small Gaussian noise
    # # no. of duplicates can be kept approx. to no. of neighbors (this also simplifies expressions on bias introduced due to duplicates)
    # # note that, duplication will increase computational cost while advantage in accuracy (due to duplication specifically) is not clear
    # # we create duplicates as per an integer valued variable "duplication multiplicity" which is ratio of no. of samples after duplication
    # #  and no. of samples before duplication
    # #considering duplicates by multiple 2 since that is minimal value required
    # #
    # min_num_samples = num_of_neighbors
    # #
    # if debug:
    #     print 'min_num_samples', min_num_samples
    # #
    # multiple_i = int(math.ceil(min_num_samples/float(n)))
    # if debug:
    #     print 'multiple_i', multiple_i
    # #
    # multiple_j = int(math.ceil(min_num_samples/float(m)))
    # if debug:
    #     print 'multiple_j', multiple_j
    # Dii = np.tile(Dii, (multiple_i, multiple_i))
    # Djj = np.tile(Djj, (multiple_j, multiple_j))
    # Dij = np.tile(Dij, (multiple_i, multiple_j))
    # n *= multiple_i
    # m *= multiple_j
    # assert m >= min_num_samples
    # assert n >= min_num_samples
    # #
    # const_epsillon = 1e-2
    # Dii += const_epsillon*(np.random.standard_normal(size=Dii.shape)**2)
    # Djj += const_epsillon*(np.random.standard_normal(size=Djj.shape)**2)
    # Dij += const_epsillon*(np.random.standard_normal(size=Dij.shape)**2)
    #
    kl_i_j, l, k = eval_kl_div_knn_frm_dist(Dii, Dij, num_of_neighbors)
    kl_j_i, l_, k_ = eval_kl_div_knn_frm_dist(Djj, Dij.transpose(), num_of_neighbors)
    kl = kl_i_j + kl_j_i
    #
    if kl < 0:
        if -div_tol < kl < 0:
            kl = 0
        else:
            print '({},{})'.format(num_of_neighbors, kl)
            if debug:
                print 'kl_i_j', kl_i_j
                print 'kl_j_i', kl_j_i
                print 'Dii', Dii
                print 'Djj', Djj
                print 'Dij', Dij
            if algo_special_cond == 'kld_kd':
                return eval_max_mean_discrepancy(Kii, Kjj, Kij)
            elif algo_special_cond == 'mmd':
                return eval_max_mean_discrepancy(Kii, Kjj, Kij)
    return kl


if __name__ == '__main__':
    #
    import sys
    is_amr_only = bool(sys.argv[1])
    is_dep_only = bool(sys.argv[2])
    assert not (is_amr_only and is_dep_only)
    #
    is_load_amr_data = True
    has_data_only_amrs = False
    amr_pickle_file_path = './amr_data_temp.pickle'
    if not is_load_amr_data:
        import train_extractor as te
        if is_alternative_data:
            amr_graphs, labels = \
                te.get_data_joint(is_train=None, is_word_vectors=False, is_dependencies=(not has_data_only_amrs), is_alternative_data=True)
        else:
            amr_graphs, labels = te.get_data_joint(is_train=True, is_word_vectors=False, is_dependencies=(not has_data_only_amrs))
        with open(cap.absolute_path+amr_pickle_file_path, 'wb') as f_p:
            amr_data = {'amr': amr_graphs, 'label': labels}
            pickle.dump(amr_data, f_p)
    else:
        with open(cap.absolute_path+amr_pickle_file_path, 'rb') as f_p:
            amr_data = pickle.load(f_p)
            amr_graphs = amr_data['amr']
            labels = amr_data['label']
    #
    n = labels.size
    print 'labels.size', labels.size
    #
    import compute_parallel_graph_kernel_matrix_joint_train_data as cpgkmjtd
    K = cpgkmjtd.join_parallel_computed_kernel_matrices(217)
    print 'K.shape', K.shape
    #
    import config_kernel_matrices_format as ckmf
    if ckmf.is_kernel_dtype_lower_precision:
        K = K.astype(ckmf.kernel_dtype_np, copy=False)
    #
    eval_divergence_matrix(amr_graphs, labels, K, is_amr_only=is_amr_only, is_dep_only=is_dep_only, has_data_only_amrs=has_data_only_amrs)

