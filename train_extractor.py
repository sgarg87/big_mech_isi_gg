import gen_extractor_features_data as gtd
import time
import numpy as np
import graph_kernels as gk
from kernel_tuned_parameters import *
import sklearn.svm as skl_svm
import sklearn.metrics as skm
import extract_from_amr_dot as ead
from config import *
import sklearn.preprocessing as skp
import pickle as p
import constant_amrs_train_extractor as cate
import constants_trained_svm_kernel_file_paths as ctskfp
import constants_joint_synthetic as cjs
import constants_absolute_path as cap
import config_hpcc as ch
import parallel_kernel_eval as pke
import node_role_types as nrt
import config_darpa as cd
import protein_state_subgraph_synthetic_edges as pssse
from config_console_output import *
import eval_divergence_frm_kernel as edk
import config_train_extractor as cte
import config_kernel as ck
import pydot as pd
import file_paths_extraction as fpe
import stanford_dependencies as sd
import semi_automated_extraction_features_data as saefd
import config_processing as cp
import file_paths_train_data as fptd
import config_gp_classifier_parameters as cgcp
import scipy.sparse.linalg as ssl
import compute_parallel_graph_kernel_matrix_joint_train_data as cpgkmjtd


def is_amr_cyclic_undirected(nodes_list):
    n = len(nodes_list)
    visited_nodes = [False]*n
    try:
        while not all(visited_nodes):
            curr_root_node = nodes_list[visited_nodes.index(False)]
            dfs_undirected(curr_root_node, nodes_list, visited_nodes)
    except AssertionError:
        return True
    return False


def dfs_undirected(curr_node, nodes_list, visited_nodes, parent_node=None):
    curr_node_idx = nodes_list.index(curr_node)
    visited_nodes[curr_node_idx] = True
    for child in curr_node.create_undirected_children_list():
        if child in nodes_list: #otherwise consider child to be nonexistent for this graph
            child_idx = nodes_list.index(child)
            if not visited_nodes[child_idx]:
                dfs_undirected(child, nodes_list, visited_nodes, curr_node)
            else:
                if child != parent_node:
                    raise AssertionError


def is_amr_cyclic(nodes_list):
    n = len(nodes_list)
    counter = 0
    visited_nodes = [False]*n
    arrival_time_nodes = [-1]*n
    departure_time_nodes = [-1]*n
    try:
        while not all(visited_nodes):
            curr_root_node = nodes_list[visited_nodes.index(False)]
            counter = dfs(curr_root_node, nodes_list, counter, visited_nodes, arrival_time_nodes, departure_time_nodes)
    except AssertionError:
        return True
    return False


def dfs(curr_node, nodes_list, counter, visited_nodes, arrival_time_nodes, departure_time_nodes):
    curr_node_idx = nodes_list.index(curr_node)
    counter += 1
    arrival_time_nodes[curr_node_idx] = counter
    visited_nodes[curr_node_idx] = True
    for child in curr_node.create_children_list():
        if child in nodes_list: #otherwise consider child to be nonexistent for this graph
            child_idx = nodes_list.index(child)
            if not visited_nodes[child_idx]:
                counter = dfs(child, nodes_list, counter, visited_nodes, arrival_time_nodes, departure_time_nodes)
    counter += 1
    departure_time_nodes[curr_node_idx] = counter
    for child in curr_node.create_children_list():
        if child in nodes_list: #otherwise consider child to be nonexistent for this graph
            child_idx = nodes_list.index(child)
            if visited_nodes[child_idx]:
                if departure_time_nodes[curr_node_idx] < departure_time_nodes[child_idx] or (departure_time_nodes[child_idx] == -1 and departure_time_nodes[curr_node_idx] != -1):
                    raise AssertionError #cycle exists in the graph (back edge existence test)
    return counter


def get_map_frm_list(nodes_list):
    if debug:
        print 'processing nodes list for generating the corresponding map ...'
    nodes_map = {}
    for node in nodes_list:
        nodes_map[node.id] = node
    nodes_map['root'] = nodes_list[0]
    if debug:
        print 'root node is ', nodes_map['root']
    return nodes_map


def get_data(is_train=True):
    start_time = time.time()
    if is_train:
        data = gtd.load_pickled_merged_data(is_train=True)
    else:
        data = gtd.load_pickled_merged_data(is_train=False)
    graph_keys = data['paths_map'].keys()
    graph_keys.sort()
    n = len(graph_keys)
    print 'Number of graphs: ', n
    amr_graphs = []
    if len(data['catalyst_labels_map'].values()) != n:
        raise AssertionError
    elif len(data['domain_labels_map'].values()) != n:
        raise AssertionError
    elif len(data['state_label_map'].values()) != n:
        raise AssertionError
    is_catalyst_labels = []
    is_domain_labels = []
    is_state_labels = []
    for i in range(n):
        curr_graph_key = graph_keys[i]
        curr_nodes_list = data['paths_map'][curr_graph_key]
        curr_nodes_list = ead.remove_duplicates_based_on_ids(curr_nodes_list)
        if debug:
            print 'curr_graph_key ', curr_graph_key
        if debug:
            print 'before pruning ...'
            for node_temp in curr_nodes_list:
                print 'node_temp', node_temp
        curr_nodes_list = ead.prune_non_path_nodes_references_frm_subgraph(curr_nodes_list)
        if debug:
            print 'after pruning'
            for node_temp in curr_nodes_list:
                print 'node_temp', node_temp
        if (not is_amr_cyclic(curr_nodes_list) and not ck.is_neighbor_kernel) \
                or (not is_amr_cyclic_undirected(curr_nodes_list) and ck.is_neighbor_kernel):
            if ck.is_inverse_centralize_amr:
                if debug:
                    print 'performing concept centralization on ', curr_graph_key
                curr_nodes_list = ead.centralize_amr_at_root_node(curr_nodes_list)
                if not ch.is_hpcc:
                    ead.nodes_to_dot(curr_nodes_list, curr_graph_key+'_cc')
            amr_graphs.append({'path': curr_graph_key, 'nodes': get_map_frm_list(curr_nodes_list)})
            is_catalyst_labels.append(data['catalyst_labels_map'][curr_graph_key])
            is_domain_labels.append(data['domain_labels_map'][curr_graph_key])
            is_state_labels.append(data['state_label_map'][curr_graph_key])
        else:
            print 'Graph {} is cyclic'.format(curr_graph_key)
    n = len(amr_graphs)
    print 'Number of graphs after filtering of cyclic ones : ', n
    amr_graphs = np.array(amr_graphs, dtype=np.object).reshape((n, 1))
    is_catalyst_labels = np.array(is_catalyst_labels)
    is_domain_labels = np.array(is_domain_labels)
    is_state_labels = np.array(is_state_labels)
    if ck.is_word_vectors:
        print 'Preprocessing AMR graphs for assigning word vectors ...'
        start_time = time.time()
        for i in range(n):
            gk.preprocess_amr_fr_assign_wordvector(amr_graphs[i, 0])
        print 'Execution time to assign vectors for each node in AMR graphs was ', time.time()-start_time
    return amr_graphs, is_catalyst_labels, is_domain_labels, is_state_labels


def find_node_fr_id_in_list_of_nodes(nodes_list, id):
    for curr_node in nodes_list:
        if curr_node.id == id:
            return curr_node


def find_node_fr_color_in_list_of_nodes(nodes_list, color_code):
    assert color_code is not None
    for curr_node in nodes_list:
        if curr_node.color == color_code:
            return curr_node


def add_synthetic_edge_joint_subgraph(curr_nodes_list_graph, curr_triplet_nodes_tuple):
    assert curr_triplet_nodes_tuple is not None and curr_triplet_nodes_tuple
    assert curr_nodes_list_graph is not None and curr_nodes_list_graph
    assert 3 <= len(curr_triplet_nodes_tuple) <= 4
    concept_node = curr_triplet_nodes_tuple[0]
    catalyst_node = curr_triplet_nodes_tuple[1]
    protein_node = curr_triplet_nodes_tuple[2]
    if len(curr_triplet_nodes_tuple) == 4:
        protein_node2 = curr_triplet_nodes_tuple[3]
    else:
        protein_node2 = None
    # getting concept node from the graph
    concept_node_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, concept_node.id)
    if debug:
        print 'concept_node_in_graph', concept_node_in_graph
    assert concept_node_in_graph is not None
    protein_node_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, protein_node.id)
    if debug:
        print 'protein_node_in_graph', protein_node_in_graph
    assert protein_node_in_graph is not None
    if catalyst_node is not None:
        catalyst_node_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, catalyst_node.id)
        if debug:
            print 'catalyst_node_in_graph', catalyst_node_in_graph
        assert catalyst_node_in_graph is not None
    else:
        catalyst_node_in_graph = None
    if protein_node2 is not None:
        protein_node2_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, protein_node2.id)
        if debug:
            print 'protein_node2_in_graph', protein_node2_in_graph
        assert protein_node2_in_graph is not None
    else:
        protein_node2_in_graph = None
    #
    concept_node = None
    del concept_node
    catalyst_node = None
    del catalyst_node
    protein_node = None
    del protein_node
    protein_node2 = None
    del protein_node2
    #
    if catalyst_node_in_graph is not None:
        is_catalyst_parent_of_concept, _ = concept_node_in_graph.is_parent(catalyst_node_in_graph)
        if is_catalyst_parent_of_concept:
            catalyst_node_in_graph.add_parent_child_relationship(concept_node_in_graph, cjs.hasCatalyst+'-of')
        else:
            concept_node_in_graph.add_parent_child_relationship(catalyst_node_in_graph, cjs.hasCatalyst)
    is_protein_parent_of_concept, _ = concept_node_in_graph.is_parent(protein_node_in_graph)
    if is_protein_parent_of_concept:
        protein_node_in_graph.add_parent_child_relationship(concept_node_in_graph, cjs.hasProtein+'-of')
    else:
        concept_node_in_graph.add_parent_child_relationship(protein_node_in_graph, cjs.hasProtein)
    if protein_node2_in_graph is not None:
        is_protein2_parent_of_concept, _ = concept_node_in_graph.is_parent(protein_node2_in_graph)
        if is_protein2_parent_of_concept:
            protein_node2_in_graph.add_parent_child_relationship(concept_node_in_graph, cjs.hasProtein2+'-of')
        else:
            concept_node_in_graph.add_parent_child_relationship(protein_node2_in_graph, cjs.hasProtein2)


def get_protein_concept_state_nodes_wd_ids_from_subgraph(curr_nodes_list_graph, curr_triplet_nodes_tuple):
    assert curr_triplet_nodes_tuple is not None and curr_triplet_nodes_tuple
    assert curr_nodes_list_graph is not None and curr_nodes_list_graph
    assert len(curr_triplet_nodes_tuple) == 3
    protein_node = curr_triplet_nodes_tuple[0]
    concept_node = curr_triplet_nodes_tuple[1]
    state_node = curr_triplet_nodes_tuple[2]
    #
    concept_node = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, concept_node.id)
    if debug:
        print 'concept_node_in_graph', concept_node
    assert concept_node is not None
    #
    protein_node = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, protein_node.id)
    if debug:
        print 'protein_node_in_graph', protein_node
    assert protein_node is not None
    #
    state_node = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, state_node.id)
    if debug:
        print 'state_node_in_graph', state_node
    assert state_node is not None
    return protein_node, concept_node, state_node


def remove_synthetic_edge_protein_state_sub_graph(curr_nodes_list_graph, curr_triplet_nodes_tuple):
    protein_node, concept_node, state_node = get_protein_concept_state_nodes_wd_ids_from_subgraph(
        curr_nodes_list_graph, curr_triplet_nodes_tuple)
    #
    if pssse.relatedToConcept in protein_node.children:
        protein_node.remove_parent_child_relationship(concept_node, pssse.relatedToConcept)
    elif pssse.relatedToConcept_of in concept_node.children:
        concept_node.remove_parent_child_relationship(protein_node, pssse.relatedToConcept_of)
    else:
        raise AssertionError
    #
    if pssse.hasState in protein_node.children:
        protein_node.remove_parent_child_relationship(state_node, pssse.hasState)
    elif pssse.hasState_of in state_node.children:
        state_node.remove_parent_child_relationship(protein_node, pssse.hasState_of)
    else:
        raise AssertionError


def get_triplet_nodes_tuple_frm_joint_subgraph(curr_nodes_list_graph):
    # getting concept node from the graph
    concept_node_in_graph = find_node_fr_color_in_list_of_nodes(curr_nodes_list_graph, 'blue')
    if debug:
        print 'concept_node_in_graph', concept_node_in_graph
    assert concept_node_in_graph is not None
    #
    protein_node_in_graph = find_node_fr_color_in_list_of_nodes(curr_nodes_list_graph, '#976850')
    if debug:
        print 'protein_node_in_graph', protein_node_in_graph
    assert protein_node_in_graph is not None
    #
    catalyst_node_in_graph = find_node_fr_color_in_list_of_nodes(curr_nodes_list_graph, 'green')
    if debug:
        print 'catalyst_node_in_graph', catalyst_node_in_graph
    #
    protein_node2_in_graph = find_node_fr_color_in_list_of_nodes(curr_nodes_list_graph, '#976856')
    if debug:
        print 'protein_node2_in_graph', protein_node2_in_graph
    #
    curr_triplet_nodes_tuple = [concept_node_in_graph, catalyst_node_in_graph, protein_node_in_graph]
    if protein_node2_in_graph is not None:
        curr_triplet_nodes_tuple.append(protein_node2_in_graph)
    curr_triplet_nodes_tuple = tuple(curr_triplet_nodes_tuple)
    return curr_triplet_nodes_tuple


def add_synthetic_role_joint_subgraph(curr_nodes_list_graph, curr_triplet_nodes_tuple=None):
    # assert curr_triplet_nodes_tuple is not None and curr_triplet_nodes_tuple
    assert curr_nodes_list_graph is not None and curr_nodes_list_graph
    if curr_triplet_nodes_tuple is not None:
        assert 3 <= len(curr_triplet_nodes_tuple) <= 4
        concept_node = curr_triplet_nodes_tuple[0]
        catalyst_node = curr_triplet_nodes_tuple[1]
        protein_node = curr_triplet_nodes_tuple[2]
        if len(curr_triplet_nodes_tuple) == 4:
            protein_node2 = curr_triplet_nodes_tuple[3]
        else:
            protein_node2 = None
        # getting concept node from the graph
        concept_node_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, concept_node.id)
        if debug:
            print 'concept_node_in_graph', concept_node_in_graph
        assert concept_node_in_graph is not None
        protein_node_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, protein_node.id)
        if debug:
            print 'protein_node_in_graph', protein_node_in_graph
        assert protein_node_in_graph is not None
        if catalyst_node is not None:
            catalyst_node_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, catalyst_node.id)
            if debug:
                print 'catalyst_node_in_graph', catalyst_node_in_graph
            assert catalyst_node_in_graph is not None
        else:
            catalyst_node_in_graph = None
        if protein_node2 is not None:
            protein_node2_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, protein_node2.id)
            if debug:
                print 'protein_node2_in_graph', protein_node2_in_graph
            assert protein_node2_in_graph is not None
        else:
            protein_node2_in_graph = None
        #
        concept_node = None
        del concept_node
        catalyst_node = None
        del catalyst_node
        protein_node = None
        del protein_node
        protein_node2 = None
        del protein_node2
    else:
        # getting concept node from the graph
        concept_node_in_graph = find_node_fr_color_in_list_of_nodes(curr_nodes_list_graph, 'blue')
        if debug:
            print 'concept_node_in_graph', concept_node_in_graph
        assert concept_node_in_graph is not None
        #
        protein_node_in_graph = find_node_fr_color_in_list_of_nodes(curr_nodes_list_graph, '#976850')
        if debug:
            print 'protein_node_in_graph', protein_node_in_graph
        assert protein_node_in_graph is not None
        #
        catalyst_node_in_graph = find_node_fr_color_in_list_of_nodes(curr_nodes_list_graph, 'green')
        if debug:
            print 'catalyst_node_in_graph', catalyst_node_in_graph
        #
        protein_node2_in_graph = find_node_fr_color_in_list_of_nodes(curr_nodes_list_graph, '#976856')
        if debug:
            print 'protein_node2_in_graph', protein_node2_in_graph
    #
    if catalyst_node_in_graph is not None:
        catalyst_node_in_graph.role = nrt.catalyst
    protein_node_in_graph.role = nrt.domain
    concept_node_in_graph.role = nrt.concept
    if protein_node2_in_graph is not None:
        protein_node2_in_graph.role = nrt.domain2
    if debug:
        print '************************************'
        print 'after adding roles'
        print '************************************'
        print 'concept_node_in_graph', concept_node_in_graph
        print 'catalyst_node_in_graph', catalyst_node_in_graph
        print 'protein_node_in_graph', protein_node_in_graph
        print 'protein_node2_in_graph', protein_node2_in_graph
        print '************************************'


def add_synthetic_role_protein_state_sub_graph(curr_nodes_list_graph, curr_triplet_nodes_tuple=None):
    protein_node, concept_node, state_node = get_protein_concept_state_nodes_wd_ids_from_subgraph(
        curr_nodes_list_graph, curr_triplet_nodes_tuple)
    #
    protein_node.role = nrt.protein
    concept_node.role = nrt.concept
    state_node.role = nrt.state
    if debug:
        print '************************************'
        print 'after adding roles'
        print '************************************'
        print 'protein_node', protein_node
        print 'concept_node', concept_node
        print 'state_node', state_node
        print '************************************'


def set_default_role_fr_all_nodes(curr_nodes_list):
    for curr_node in curr_nodes_list:
        curr_node.role = nrt.no_role


def eliminate_synthetic_edge_cycles(curr_nodes_list_graph, curr_protein_concept_state_nodes_tuple):
    assert curr_protein_concept_state_nodes_tuple is not None and curr_protein_concept_state_nodes_tuple
    assert curr_nodes_list_graph is not None and curr_nodes_list_graph
    protein_node = curr_protein_concept_state_nodes_tuple[0]
    concept_node = curr_protein_concept_state_nodes_tuple[1]
    state_node = curr_protein_concept_state_nodes_tuple[2]
    # getting concept node from the graph
    concept_node_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, concept_node.id)
    if debug:
        print 'concept_node_in_graph', concept_node_in_graph
    assert concept_node_in_graph is not None
    protein_node_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, protein_node.id)
    if debug:
        print 'protein_node_in_graph', protein_node_in_graph
    assert protein_node_in_graph is not None
    state_node_in_graph = find_node_fr_id_in_list_of_nodes(curr_nodes_list_graph, state_node.id)
    if debug:
        print 'state_node_in_graph', state_node_in_graph
    assert state_node_in_graph is not None
    #
    concept_node = None
    del concept_node
    state_node = None
    del state_node
    protein_node = None
    del protein_node
    #
    if protein_node_in_graph.is_parent(state_node_in_graph)[0] and state_node_in_graph.is_parent(protein_node_in_graph)[0]:
        flag, protein_state_synthetic_edge_label = state_node_in_graph.is_parent(protein_node_in_graph)
        if not flag:
            raise AssertionError
        flag = None
        protein_node_in_graph.remove_parent_child_relationship(state_node_in_graph, protein_state_synthetic_edge_label)
        state_node_in_graph.add_parent_child_relationship(protein_node_in_graph, ead.get_inverse_of_edge_label(protein_state_synthetic_edge_label))
        protein_state_synthetic_edge_label = None
    if protein_node_in_graph.is_parent(concept_node_in_graph)[0] and concept_node_in_graph.is_parent(protein_node_in_graph)[0]:
        flag, protein_concept_synthetic_edge_label = concept_node_in_graph.is_parent(protein_node_in_graph)
        if not flag:
            raise AssertionError
        protein_node_in_graph.remove_parent_child_relationship(concept_node_in_graph, protein_concept_synthetic_edge_label)
        concept_node_in_graph.add_parent_child_relationship(protein_node_in_graph, ead.get_inverse_of_edge_label(protein_concept_synthetic_edge_label))
        protein_concept_synthetic_edge_label = None


def merge_joint_model_data_to_train_data(data, model_data):
    assert data is not None
    assert model_data is not None
    #
    print 'data.keys()', data.keys()
    print 'model_data.keys()', model_data.keys()
    data[gtd.const_paths_map].update(model_data[gtd.const_paths_map])
    #
    data[gtd.const_interaction_tuples_map].update(model_data[gtd.const_interaction_tuples_map])
    #
    if gtd.const_sentences_map not in data:
        data[gtd.const_sentences_map] = {}
    data[gtd.const_sentences_map].update(model_data[gtd.const_sentences_map])
    #
    data[gtd.const_joint_labels_map].update(model_data[gtd.const_joint_labels_map])


def get_sentence_frm_amr_dot_file(amr_dot_file_path):
    amr = pd.graph_from_dot_file(cap.absolute_path+amr_dot_file_path)
    sentence = amr.get_label()
    return sentence


def process_pickled_data(data,
        is_joint_amr_synthetic_edge=ck.is_joint_amr_synthetic_edge,
        is_joint_amr_synthetic_role=ck.is_joint_amr_synthetic_role,
        is_inverse_centralize_amr=ck.is_inverse_centralize_amr,
        is_word_vectors=ck.is_word_vectors,
        is_labels=True):
    #
    if is_word_vectors:
        assert cp.is_processing_amrs
    #
    start_time = time.time()
    #
    graph_keys = data['paths_map'].keys()
    graph_keys.sort()
    n = len(graph_keys)
    print 'Number of graphs: ', n
    amr_graphs = []
    if len(data['joint_labels_map'].values()) != n:
        print 'len ...'
        print len(data['joint_labels_map'])
        print len(data['paths_map'])
        print 'There are some paths for which labels are not available. cleaning of data is required. do after the evaluation ...'
        # raise AssertionError
    if is_labels:
        labels = []
    org_amr_sentences_map = {}
    for i in range(n):
        curr_graph_key = graph_keys[i]
        #
        if curr_graph_key not in data['joint_labels_map'] and is_labels:
            continue
        #
        if debug:
            print 'curr_graph_key ', curr_graph_key
        #
        is_cyclic_curr_graph_key = False
        for curr_cyclic_amr_path in cate.cyclic_amr_list:
            if curr_cyclic_amr_path in curr_graph_key:
                is_cyclic_curr_graph_key = True
                break
        if is_cyclic_curr_graph_key:
            continue
        curr_nodes_list = data['paths_map'][curr_graph_key]
        #
        if cd.is_darpa and cd.is_darpa_entity_type:
            ead.map_entity_types_to_darpa_types(curr_nodes_list)
        #
        if curr_graph_key not in data['interaction_tuples_map']:
            curr_triplet_nodes_tuple = get_triplet_nodes_tuple_frm_joint_subgraph(curr_nodes_list_graph=curr_nodes_list)
        else:
            curr_triplet_nodes_tuple = data['interaction_tuples_map'][curr_graph_key]
        #
        if curr_graph_key not in data[gtd.const_sentences_map]:
            curr_org_amr_dot_file_path = fpe.extract_original_amr_dot_file_name(curr_graph_key)
            if curr_org_amr_dot_file_path in org_amr_sentences_map:
                curr_sentence = org_amr_sentences_map[curr_org_amr_dot_file_path]
            else:
                curr_sentence = get_sentence_frm_amr_dot_file(curr_org_amr_dot_file_path)
                org_amr_sentences_map[curr_org_amr_dot_file_path] = curr_sentence
        else:
            curr_sentence = data[gtd.const_sentences_map][curr_graph_key]
        #
        if is_joint_amr_synthetic_edge or is_joint_amr_synthetic_role:
            print 'curr_graph_key', curr_graph_key
            if is_joint_amr_synthetic_role:
                set_default_role_fr_all_nodes(curr_nodes_list)
            #
            if is_joint_amr_synthetic_edge:
                add_synthetic_edge_joint_subgraph(curr_nodes_list, curr_triplet_nodes_tuple)
            if is_joint_amr_synthetic_role:
                add_synthetic_role_joint_subgraph(curr_nodes_list, curr_triplet_nodes_tuple)
        curr_nodes_list = ead.remove_duplicates_based_on_ids(curr_nodes_list)
        curr_nodes_list = ead.prune_non_path_nodes_references_frm_subgraph(curr_nodes_list)
        if is_inverse_centralize_amr:
            if debug:
                print 'performing concept centralization on ', curr_graph_key
            curr_nodes_list = ead.centralize_amr_at_root_node(curr_nodes_list)
            if is_amr_cyclic(curr_nodes_list):
                curr_nodes_list = ead.eliminate_first_order_cycles(curr_nodes_list)
            if not ch.is_hpcc:
                ead.nodes_to_dot(curr_nodes_list, curr_graph_key+'_cc')
        else:
            curr_nodes_list = ead.eliminate_first_order_cycles(curr_nodes_list)
        if not is_amr_cyclic(curr_nodes_list):
            amr_graphs.append(
                {
                    'path': curr_graph_key,
                    'nodes': get_map_frm_list(curr_nodes_list),
                    'tuple': curr_triplet_nodes_tuple,
                    'text': curr_sentence
                })
            if is_labels:
                labels.append(data['joint_labels_map'][curr_graph_key])
        else:
            print 'Graph {} is cyclic'.format(curr_graph_key)
    n = len(amr_graphs)
    print 'Number of graphs after filtering of cyclic ones : ', n
    amr_graphs = np.array(amr_graphs, dtype=np.object).reshape((n, 1))
    if is_labels:
        labels = np.array(labels)
    if is_word_vectors:
        print 'Pre-processing AMR graphs for assigning word vectors ...'
        start_time = time.time()
        for i in range(n):
            gk.preprocess_amr_fr_assign_wordvector(amr_graphs[i, 0])
        print 'Execution time to assign vectors for each node in AMR graphs was ', time.time()-start_time
    if is_labels:
        return amr_graphs, labels
    else:
        return amr_graphs


def get_data_joint(
        is_train=True,
        is_model_data=cte.is_model_data,
        is_model_interactions_graph_in_joint_train=cte.is_model_interactions_graph_in_joint_train,
        is_non_synthetic_model_data_only=cte.is_non_synthetic_model_data_only,
        is_joint_amr_synthetic_edge=ck.is_joint_amr_synthetic_edge,
        is_joint_amr_synthetic_role=ck.is_joint_amr_synthetic_role,
        is_inverse_centralize_amr=ck.is_inverse_centralize_amr,
        is_neighbor_kernel=ck.is_neighbor_kernel,
        is_word_vectors=ck.is_word_vectors,
        is_dependencies=cte.is_dependencies,
        load_sentence_frm_dot_if_required=True,
        is_alternative_data=False,
        is_chicago_data=cte.is_chicago_data):
    #
    if is_word_vectors:
        assert cp.is_processing_amrs
    #
    start_time = time.time()
    if is_alternative_data:
        assert is_train is None
        assert is_chicago_data is None
        data = saefd.load_aimed_pickled_filtered_labeled_data_joint()
        #
        if coarse_debug:
            print '*********paths_map keys *********'
            print data['paths_map'].keys()
            print '*********joint labels keys ************'
            print data['joint_labels_map'].keys()
            print '*******joint labels map ************'
            print data['joint_labels_map']
            print '**********************'
        #
        if is_dependencies:
            dependencies_data = saefd.load_aimed_pickled_filtered_labeled_sdg_data_joint()
            merge_joint_model_data_to_train_data(data, dependencies_data)
    else:
        assert is_train is not None
        if is_train:
            if is_non_synthetic_model_data_only:
                data = gtd.load_pickled_joint_data_model(is_synthetic=False)
            else:
                data = gtd.load_pickled_merged_data(is_train=True)
                if coarse_debug:
                    print '*********paths_map keys *********'
                    print data['paths_map'].keys()
                    print '*********joint labels keys ************'
                    print data['joint_labels_map'].keys()
                    print '*******joint labels map ************'
                    print data['joint_labels_map']
                    print '**********************'
                #
                if is_chicago_data:
                    chicago_data = saefd.load_chicago_data_joint()
                    merge_joint_model_data_to_train_data(data, chicago_data)
                #
                if is_model_data:
                    model_data_non_synthetic = gtd.load_pickled_joint_data_model(is_synthetic=False)
                    merge_joint_model_data_to_train_data(data, model_data_non_synthetic)
            if is_model_interactions_graph_in_joint_train:
                assert is_model_data
                model_data = gtd.load_pickled_joint_data_model(is_synthetic=True)
                merge_joint_model_data_to_train_data(data, model_data)
            if is_dependencies:
                sd_obj = sd.StanfordDependencies()
                dependencies_data = sd_obj.load_stanford_dependencies_train_data()
                merge_joint_model_data_to_train_data(data, dependencies_data)
                #
                if is_chicago_data:
                    chicago_data_sdg = saefd.load_chicago_sdg_data_joint()
                    merge_joint_model_data_to_train_data(data, chicago_data_sdg)
        else:
            assert is_chicago_data is None
            data = gtd.load_pickled_merged_data(is_train=False)
            if is_dependencies:
                raise NotImplementedError('no test data in dependencies subgraphs')
    #
    graph_keys = data['paths_map'].keys()
    graph_keys.sort()
    n = len(graph_keys)
    print 'Number of graphs: ', n
    amr_graphs = []
    if len(data['joint_labels_map'].values()) != n:
        print 'len ...'
        print len(data['joint_labels_map'])
        print len(data['paths_map'])
        print 'There are some paths for which labels are not available. cleaning of data is required. do after the evaluation ...'
        # raise AssertionError
    labels = []
    org_amr_sentences_map = {}
    for i in range(n):
        curr_graph_key = graph_keys[i]
        #
        if coarse_debug:
            print 'curr_graph_key ', curr_graph_key
        #
        if curr_graph_key not in data['joint_labels_map']:
            print 'skipping this one since it does not have a label'
            continue
        if not is_neighbor_kernel:
            is_cyclic_curr_graph_key = False
            for curr_cyclic_amr_path in cate.cyclic_amr_list:
                if curr_cyclic_amr_path in curr_graph_key:
                    is_cyclic_curr_graph_key = True
                    break
            if is_cyclic_curr_graph_key:
                continue
        curr_nodes_list = data['paths_map'][curr_graph_key]
        if cd.is_darpa:
            ead.map_entity_types_to_darpa_types(curr_nodes_list)
        #
        if curr_graph_key not in data['interaction_tuples_map']:
            curr_triplet_nodes_tuple = get_triplet_nodes_tuple_frm_joint_subgraph(curr_nodes_list_graph=curr_nodes_list)
        else:
            curr_triplet_nodes_tuple = data['interaction_tuples_map'][curr_graph_key]
        if curr_graph_key not in data[gtd.const_sentences_map]:
            if load_sentence_frm_dot_if_required:
                curr_org_amr_dot_file_path = fpe.extract_original_amr_dot_file_name(curr_graph_key)
                if curr_org_amr_dot_file_path in org_amr_sentences_map:
                    curr_sentence = org_amr_sentences_map[curr_org_amr_dot_file_path]
                else:
                    curr_sentence = get_sentence_frm_amr_dot_file(curr_org_amr_dot_file_path)
                    org_amr_sentences_map[curr_org_amr_dot_file_path] = curr_sentence
            else:
                curr_sentence = None
        else:
            curr_sentence = data[gtd.const_sentences_map][curr_graph_key]
        #
        if is_joint_amr_synthetic_edge or is_joint_amr_synthetic_role:
            if coarse_debug:
                print 'curr_graph_key', curr_graph_key
            if is_joint_amr_synthetic_role:
                set_default_role_fr_all_nodes(curr_nodes_list)
            # if curr_graph_key in data['interaction_tuples_map']:
            if is_train and is_model_interactions_graph_in_joint_train and curr_graph_key in model_data['paths_map']:
                if is_joint_amr_synthetic_role:
                    raise NotImplementedError
            else:
                if is_joint_amr_synthetic_edge:
                    add_synthetic_edge_joint_subgraph(curr_nodes_list, curr_triplet_nodes_tuple)
                if is_joint_amr_synthetic_role:
                    add_synthetic_role_joint_subgraph(curr_nodes_list, curr_triplet_nodes_tuple)
            # else:
            #     if is_joint_amr_synthetic_role:
            #         if is_train and is_model_interactions_graph_in_joint_train and curr_graph_key in model_data['paths_map']:
            #             raise NotImplementedError
            #         add_synthetic_role_joint_subgraph(curr_nodes_list)
        curr_nodes_list = ead.remove_duplicates_based_on_ids(curr_nodes_list)
        curr_nodes_list = ead.prune_non_path_nodes_references_frm_subgraph(curr_nodes_list)
        if is_inverse_centralize_amr:
            if debug:
                print 'performing concept centralization on ', curr_graph_key
            curr_nodes_list = ead.centralize_amr_at_root_node(curr_nodes_list)
            if (is_amr_cyclic(curr_nodes_list) and not is_neighbor_kernel) \
                    or (is_amr_cyclic_undirected(curr_nodes_list) and is_neighbor_kernel):
                curr_nodes_list = ead.eliminate_first_order_cycles(curr_nodes_list)
            if not ch.is_hpcc:
                ead.nodes_to_dot(curr_nodes_list, curr_graph_key+'_cc')
        else:
            curr_nodes_list = ead.eliminate_first_order_cycles(curr_nodes_list)
        if (not is_amr_cyclic(curr_nodes_list) and not is_neighbor_kernel) \
                or (not is_amr_cyclic_undirected(curr_nodes_list) and is_neighbor_kernel):
            amr_graphs.append(
                {
                    'path': curr_graph_key,
                    'nodes': get_map_frm_list(curr_nodes_list),
                    'tuple': curr_triplet_nodes_tuple,
                    'text': curr_sentence
                })
            labels.append(data['joint_labels_map'][curr_graph_key])
        else:
            print 'Graph {} is cyclic'.format(curr_graph_key)
    n = len(amr_graphs)
    print 'Number of graphs after filtering of cyclic ones : ', n
    amr_graphs = np.array(amr_graphs, dtype=np.object).reshape((n, 1))
    # np.save(cap.absolute_path+'./temp_amr_graphs', amr_graphs)
    labels = np.array(labels)
    if is_word_vectors:
        print 'Preprocessing AMR graphs for assigning word vectors ...'
        start_time = time.time()
        for i in range(n):
            gk.preprocess_amr_fr_assign_wordvector(amr_graphs[i, 0])
        print 'Execution time to assign vectors for each node in AMR graphs was ', time.time()-start_time
    return amr_graphs, labels


def get_data_joint_dependencies(is_train=True):
    if not is_train:
        raise NotImplementedError('no stanford dependencies generated for test (also not required)')



def change_root_node(nodes_list, root_node_id):
    is_root_node_id_found = False
    for curr_node in nodes_list:
        if curr_node.id == root_node_id:
            if is_root_node_id_found:
                raise AssertionError
            curr_node.is_root = True
            is_root_node_id_found = True
        else:
            curr_node.is_root = False
    if not is_root_node_id_found:
        raise AssertionError


def merge_protein_state_model_data_to_train_data(data, model_data):
    assert data is not None
    assert model_data is not None
    #
    print 'data.keys', data.keys()
    print 'model_data.keys', model_data.keys()
    #
    data[gtd.const_paths_map].update(model_data[gtd.const_paths_map])
    #
    data[gtd.const_protein_state_tuples_map].update(model_data[gtd.const_protein_state_tuples_map])
    #
    data[gtd.const_sentences_map].update(model_data[gtd.const_sentences_map])
    #
    data[gtd.const_joint_labels_map].update(model_data[gtd.const_joint_labels_map])


def get_protein_state_data(is_train=True):
    start_time = time.time()
    if is_train:
        data = gtd.load_pickled_protein_state_data(is_train=is_train)
        if cte.is_model_interactions_graph_in_protein_state_train:
            model_data = gtd.load_pickled_protein_state_data_model()
            merge_protein_state_model_data_to_train_data(data, model_data)
    else:
        raise NotImplementedError
    graph_keys = data['paths_map'].keys()
    graph_keys.sort()
    n = len(graph_keys)
    print 'Number of graphs: ', n
    amr_graphs = []
    if len(data['joint_labels_map'].values()) != n:
        raise AssertionError
    labels = []
    for i in range(n):
        curr_graph_key = graph_keys[i]
        print 'curr_graph_key', curr_graph_key
        if not ck.is_neighbor_kernel:
            is_cyclic_curr_graph_key = False
            for curr_cyclic_amr_path in cate.cyclic_amr_list:
                if curr_cyclic_amr_path in curr_graph_key:
                    is_cyclic_curr_graph_key = True
                    break
            if is_cyclic_curr_graph_key:
                continue
        #
        curr_protein_concept_state_nodes_tuple = data['protein_state_tuples_map'][curr_graph_key]
        curr_nodes_list = data['paths_map'][curr_graph_key]
        if cd.is_darpa:
            ead.map_entity_types_to_darpa_types(curr_nodes_list)
        if (not ck.is_protein_state_amr_synthetic_edge) or ck.is_protein_state_amr_synthetic_role:
            print 'curr_graph_key', curr_graph_key
            if is_train and cte.is_model_interactions_graph_in_joint_train and curr_graph_key in model_data['paths_map']:
                    raise NotImplementedError
            if ck.is_protein_state_amr_synthetic_role:
                set_default_role_fr_all_nodes(curr_nodes_list)
            if not ck.is_protein_state_amr_synthetic_edge:
                remove_synthetic_edge_protein_state_sub_graph(curr_nodes_list, curr_protein_concept_state_nodes_tuple)
            if ck.is_protein_state_amr_synthetic_role:
                add_synthetic_role_protein_state_sub_graph(curr_nodes_list, curr_protein_concept_state_nodes_tuple)
        #
        curr_nodes_list = ead.remove_duplicates_based_on_ids(curr_nodes_list)
        curr_nodes_list = ead.prune_non_path_nodes_references_frm_subgraph(curr_nodes_list)
        if ck.is_protein_state_amr_synthetic_edge:
            eliminate_synthetic_edge_cycles(curr_nodes_list, curr_protein_concept_state_nodes_tuple)
        if ck.is_inverse_centralize_amr:
            if debug:
                print 'performing concept centralization on ', curr_graph_key
            if ck.is_protein_state_subgraph_rooted_at_concept_node:
                change_root_node(curr_nodes_list, curr_protein_concept_state_nodes_tuple[1].id)
                curr_nodes_list = ead.centralize_amr_at_root_node(curr_nodes_list, curr_protein_concept_state_nodes_tuple[1].id)
            else:
                curr_nodes_list = ead.centralize_amr_at_root_node(curr_nodes_list)
            if (is_amr_cyclic(curr_nodes_list) and not ck.is_neighbor_kernel) \
                    or (is_amr_cyclic_undirected(curr_nodes_list) and ck.is_neighbor_kernel):
                curr_nodes_list = ead.eliminate_first_order_cycles(curr_nodes_list)
            if not ch.is_hpcc:
                ead.nodes_to_dot(curr_nodes_list, curr_graph_key+'_cc')
        else:
            curr_nodes_list = ead.eliminate_first_order_cycles(curr_nodes_list)
        if debug:
            print 'curr_graph_key ', curr_graph_key
        if (not is_amr_cyclic(curr_nodes_list) and not ck.is_neighbor_kernel) or (not is_amr_cyclic_undirected(curr_nodes_list)
                                                                                  and ck.is_neighbor_kernel):
            amr_graphs.append({'path': curr_graph_key, 'nodes': get_map_frm_list(curr_nodes_list)})
            labels.append(data['joint_labels_map'][curr_graph_key])
        else:
            print 'Graph {} is cyclic'.format(curr_graph_key)
    n = len(amr_graphs)
    print 'Number of graphs after filtering of cyclic ones : ', n
    amr_graphs = np.array(amr_graphs, dtype=np.object).reshape((n, 1))
    labels = np.array(labels)
    if ck.is_word_vectors:
        print 'Preprocessing AMR graphs for assigning word vectors ...'
        start_time = time.time()
        for i in range(n):
            gk.preprocess_amr_fr_assign_wordvector(amr_graphs[i, 0])
        print 'Execution time to assign vectors for each node in AMR graphs was ', time.time()-start_time
    return amr_graphs, labels


def tune_classification(amr_graphs, labels):
    opt_ct = None
    if ck.is_word_vectors:
        opt_ct, opt_lam, opt_lam_ars = gk.tune_svm_kernel_cosine_threshold(amr_graphs, labels, ck.ct_range_min, ck.ct_range_max)
        print 'Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is ', opt_ct
    else:
        opt_lam, opt_lam_ars = gk.tune_svm_kernel_lambda(amr_graphs, labels, ck.lambda_range_min, ck.lambda_range_max)
    print 'Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is ', opt_lam
    print 'Mean adjusted rand score for optimal lambda is ', opt_lam_ars['mean']
    print 'Max adjusted rand score for optimal lambda is ', opt_lam_ars['max']
    print 'Min adjusted rand score for optimal lambda is ', opt_lam_ars['min']
    print 'SD adjusted rand score for optimal lambda is ', opt_lam_ars['std']
    return opt_ct, opt_lam, opt_lam_ars


def get_processed_train_joint_data():
    with open(cap.absolute_path+fptd.processed_amr_graphs_lables_joint_train, 'rb') as f_processed_amr_graphs_lables_joint_train:
        start_time = time.time()
        print 'loading ...'
        data = p.load(f_processed_amr_graphs_lables_joint_train)
        print 'loaded in time ', time.time()-start_time
        amr_graphs = data['amr']
        labels = data['label']
        data = None
    #
    return amr_graphs, labels


def build_extraction_classifier(is_tuning_clf=True):
    def save_nd_plot(K, matrix_name):
        if ck.is_sparse:
            matrix_name += '_sparse'
        np.save(matrix_name, K)
        #plotting
        # plt.pcolor(K)
        # plt.colorbar()
        # plt.savefig(matrix_name+'.pdf', dpi=1000)
        # plt.close()

    if is_tuning_clf:
        if not cte.is_joint:
            amr_graphs, is_catalyst_labels, is_domain_labels, _ = get_data()
        else:
            amr_graphs, labels = get_data_joint()
    if not cte.is_multiclass:
        if is_tuning_clf:
            #catalyst
            opt_ct_catalyst, opt_lam_catalyst, opt_lam_ars_catalyst = tune_classification(amr_graphs, is_catalyst_labels)
            K_catalyst = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=opt_lam_catalyst, cosine_threshold=opt_ct_catalyst)
            save_nd_plot(K_catalyst, 'K_catalyst')
            #domain
            opt_ct_domain, opt_lam_domain, opt_lam_ars_domain = tune_classification(amr_graphs, is_domain_labels)
            K_domain = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=opt_lam_domain, cosine_threshold=opt_ct_domain)
            save_nd_plot(K_domain, 'K_domain')
            #printing final results
            #catalyst
            print 'Catalyst classification .....................................'
            print 'Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is ', opt_ct_catalyst
            print 'Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is ', opt_lam_catalyst
            print 'Mean adjusted rand score for optimal lambda is ', opt_lam_ars_catalyst['mean']
            print 'Max adjusted rand score for optimal lambda is ', opt_lam_ars_catalyst['max']
            print 'Min adjusted rand score for optimal lambda is ', opt_lam_ars_catalyst['min']
            print 'SD adjusted rand score for optimal lambda is ', opt_lam_ars_catalyst['std']
            #domain
            print 'Domain classification .......................................'
            print 'Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is ', opt_ct_domain
            print 'Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is ', opt_lam_domain
            print 'Mean adjusted rand score for optimal lambda is ', opt_lam_ars_domain['mean']
            print 'Max adjusted rand score for optimal lambda is ', opt_lam_ars_domain['max']
            print 'Min adjusted rand score for optimal lambda is ', opt_lam_ars_domain['min']
            print 'SD adjusted rand score for optimal lambda is ', opt_lam_ars_domain['std']
            return K_catalyst, opt_ct_catalyst, opt_lam_catalyst, K_domain, opt_ct_domain, opt_lam_domain
        else:
            raise NotImplementedError
    else:
        if cte.is_joint:
            if is_tuning_clf:
                opt_ct, opt_lam, opt_lam_ars = tune_classification(amr_graphs, labels)
                K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=opt_lam, cosine_threshold=opt_ct)
                save_nd_plot(K, 'K_concept_joint')
                #printing final results
                print 'Concept joint classification .....................................'
                print 'Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is ', opt_ct
                print 'Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is ', opt_lam
                print 'Mean adjusted rand score for optimal lambda is ', opt_lam_ars['mean']
                print 'Max adjusted rand score for optimal lambda is ', opt_lam_ars['max']
                print 'Min adjusted rand score for optimal lambda is ', opt_lam_ars['min']
                print 'SD adjusted rand score for optimal lambda is ', opt_lam_ars['std']
                return K, opt_ct, opt_lam
            else:
                amr_graphs, labels = get_processed_train_joint_data()
                # calculating kernel normalization for training points
                print 'computing normalization constant for the training points'
                num_amr = amr_graphs.shape[0]
                assert amr_graphs.shape[1] == 1
                kernel_normalization = -1*np.ones(num_amr)
                for i in range(num_amr):
                    start_time = time.time()
                    kernel_normalization[i] \
                        = gk.graph_kernel_wrapper(nodes1=amr_graphs[i, 0]['nodes'],
                                                  nodes2=amr_graphs[i, 0]['nodes'],
                                                  lam=tuned_lambda_fr_joint,
                                                  cosine_threshold=tuned_cs_fr_joint,
                                                  is_root_kernel=ck.is_root_kernel_default)
                    print 'kernel_normalization[i]', kernel_normalization[i]
                    print 'computed the normalization in ', time.time()-start_time
                assert np.all(kernel_normalization >= 0)
                #
                K = cpgkmjtd.join_parallel_computed_kernel_matrices_sparse(160)
                print 'Learning Gaussian process classifier weights ...'
                assert np.all(labels == 1), 'all labels are positive'
                score = cgcp.c*np.ones(labels.shape)
                print 'score', score
                labels = None
                start_time = time.time()
                print 'computing the least squares'
                assert K.shape[0] == K.shape[1]
                # todo: this lsqr algorithm is parallelizable, so do the needful
                # see the classical paper LSQR An algrithm for sparse linear equations and sparse least squares.pdf
                gp_weights = ssl.lsqr(K, (score-cgcp.bias), show=True)[0]
                K = None
                print 'gp_weights', gp_weights
                print 'Time to compute the least square solution was ', time.time()-start_time
                #
                trained_clf = {}
                trained_clf['model'] = 'gp'
                trained_clf['gp_weights'] = gp_weights
                trained_clf['kernel_normalization'] = kernel_normalization
                trained_clf['train_amrs'] = amr_graphs
                trained_clf['parameters'] = {}
                trained_clf['parameters']['lambda'] = tuned_lambda_fr_joint
                trained_clf['parameters']['cs'] = tuned_cs_fr_joint
                trained_clf['parameters']['gp_c'] = cgcp.c
                trained_clf['parameters']['gp_bias'] = cgcp.bias
                trained_clf['parameters']['is_root_kernel_default'] = ck.is_root_kernel_default
                trained_clf['parameters']['is_inverse_centralize_amr'] = ck.is_inverse_centralize_amr
                trained_clf['parameters']['is_joint_amr_synthetic_edge'] = ck.is_joint_amr_synthetic_edge
                trained_clf['parameters']['is_joint_amr_synthetic_role'] = ck.is_joint_amr_synthetic_role
                print 'trained_clf', trained_clf
                with open(cap.absolute_path+ctskfp.file_name_svm_classifier_multiclass_joint + '.pickle', 'wb') as h:
                    p.dump(trained_clf, h)
        else:
            if is_tuning_clf:
                #merge catalyst labels and domain labels
                labels = merge_catalyst_domain_labels(is_catalyst_labels, is_domain_labels)
                opt_ct, opt_lam, opt_lam_ars = tune_classification(amr_graphs, labels)
                K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=opt_lam, cosine_threshold=opt_ct)
                save_nd_plot(K, 'K_catalyst_domain')
                #printing final results
                print 'Catalyst-Domain-None classification .....................................'
                print 'Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is ', opt_ct
                print 'Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is ', opt_lam
                print 'Mean adjusted rand score for optimal lambda is ', opt_lam_ars['mean']
                print 'Max adjusted rand score for optimal lambda is ', opt_lam_ars['max']
                print 'Min adjusted rand score for optimal lambda is ', opt_lam_ars['min']
                print 'SD adjusted rand score for optimal lambda is ', opt_lam_ars['std']
                return K, opt_ct, opt_lam
            else:
                amr_graphs_train, is_catalyst_labels_train, is_domain_labels_train, _ = get_data(is_train=True)
                num_train = amr_graphs_train.shape[0]
                #merge two arrays
                amr_graphs = np.empty(dtype=np.object, shape=(num_train, 1))
                amr_graphs[:, 0] = amr_graphs_train[:, 0]
                svm_clf = skl_svm.SVC(kernel='precomputed', probability=True, verbose=False, class_weight='auto')
                labels_train = merge_catalyst_domain_labels(is_catalyst_labels_train, is_domain_labels_train)
                K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=tuned_lambda_fr_catalyst_domain, cosine_threshold=tuned_cs_fr_catalyst_domain)
                print 'Training ...'
                svm_clf.fit(K, labels_train)
                trained_clf = {}
                trained_clf['model'] = svm_clf
                trained_clf['train_amrs'] = amr_graphs
                trained_clf['parameters'] = {}
                trained_clf['parameters']['lambda'] = tuned_lambda_fr_catalyst_domain
                trained_clf['parameters']['cs'] = tuned_cs_fr_catalyst_domain
                trained_clf['parameters']['is_root_kernel_default'] = ck.is_root_kernel_default
                trained_clf['parameters']['is_inverse_centralize_amr'] = ck.is_inverse_centralize_amr
                with open(cap.absolute_path+ctskfp.file_name_svm_classifier_multiclass_not_joint + '.pickle', 'wb') as h:
                    p.dump(trained_clf, h)


def build_protein_state_extraction_classifier(is_tuning_clf=True):
    def lrn_nd_save_classifier(K, labels, amr_graphs, lam, cs, is_root, is_inverse):
        # calculating kernel normalization for training points
        print 'computing normalization constant for the training points'
        num_amr = amr_graphs.shape[0]
        assert amr_graphs.shape[1] == 1
        kernel_normalization = -1*np.ones(num_amr)
        for i in range(num_amr):
            start_time = time.time()
            kernel_normalization[i] \
                = gk.graph_kernel_wrapper(nodes1=amr_graphs[i, 0]['nodes'],
                                          nodes2=amr_graphs[i, 0]['nodes'],
                                          lam=tuned_lambda_fr_joint,
                                          cosine_threshold=tuned_cs_fr_joint,
                                          is_root_kernel=ck.is_root_kernel_default)
            print 'kernel_normalization[i]', kernel_normalization[i]
            print 'computed the normalization in ', time.time()-start_time
        assert np.all(kernel_normalization >= 0)
        #
        print 'Learning Gaussian process classifier weights ...'
        assert np.all(labels == 1), 'all labels are positive'
        score = cgcp.c*np.ones(labels.shape)
        print 'score', score
        labels = None
        #
        start_time = time.time()
        print 'computing the least squares'
        assert K.shape[0] == K.shape[1]
        #
        # todo: this lsqr algorithm is parallelizable, so do the needful
        # see the classical paper LSQR An algrithm for sparse linear equations and sparse least squares.pdf
        weights = ssl.lsqr(K, (score-cgcp.bias), show=True)[0]
        print 'weights', weights
        print 'Time to compute the least square solution was ', time.time()-start_time
        K = None
        #
        trained_clf = {}
        trained_clf['model'] = 'gp'
        trained_clf['gp_weights'] = weights
        trained_clf['kernel_normalization'] = kernel_normalization
        trained_clf['train_amrs'] = amr_graphs
        trained_clf['parameters'] = {}
        trained_clf['parameters']['lambda'] = lam
        trained_clf['parameters']['cs'] = cs
        trained_clf['parameters']['gp_c'] = cgcp.c
        trained_clf['parameters']['gp_bias'] = cgcp.bias
        trained_clf['parameters']['is_root_kernel_default'] = is_root
        trained_clf['parameters']['is_inverse_centralize_amr'] = is_inverse
        trained_clf['parameters']['is_protein_state_subgraph_rooted_at_concept_node'] =\
            ck.is_protein_state_subgraph_rooted_at_concept_node
        trained_clf['parameters']['is_protein_state_amr_synthetic_edge'] =\
            ck.is_protein_state_amr_synthetic_edge
        trained_clf['parameters']['is_protein_state_amr_synthetic_role'] =\
            ck.is_protein_state_amr_synthetic_role
        with open(cap.absolute_path+ctskfp.file_name_svm_classifier_protein_state + '.pickle', 'wb') as h:
            p.dump(trained_clf, h)

    def save_nd_plot(K, matrix_name):
        if ck.is_sparse:
            matrix_name += '_sparse'
        np.save(matrix_name, K)
        #plotting
        # plt.pcolor(K)
        # plt.colorbar()
        # plt.savefig(matrix_name+'.pdf', dpi=1000)
        # plt.close()

    amr_graphs, labels = get_protein_state_data(is_train=True)
    #
    # binary classification only
    print 'binary classification only'
    label2_idx = np.where(labels == 2)
    labels[label2_idx] = 0
    label2_idx = None
    #
    # filter the positive only labels
    print 'filtering positive only'
    positive_idx = np.where(labels == 1)[0]
    amr_graphs = amr_graphs[positive_idx, :]
    labels = labels[positive_idx]
    positive_idx = None
    print 'filtered'
    print 'amr_graphs.shape', amr_graphs.shape
    print 'labels.shape', labels.shape
    #
    if is_tuning_clf:
        raise NotImplementedError, 'classifier changed from svm to GP. Rewrite code for tuning with GP'
        opt_ct, opt_lam, opt_lam_ars = tune_classification(amr_graphs, labels)
        K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=opt_lam, cosine_threshold=opt_ct)
        save_nd_plot(K, 'K_state')
        #printing final results
        print 'Protein state classification .....................................'
        print 'Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is ', opt_ct
        print 'Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is ', opt_lam
        print 'Mean adjusted rand score for optimal lambda is ', opt_lam_ars['mean']
        print 'Max adjusted rand score for optimal lambda is ', opt_lam_ars['max']
        print 'Min adjusted rand score for optimal lambda is ', opt_lam_ars['min']
        print 'SD adjusted rand score for optimal lambda is ', opt_lam_ars['std']
        lrn_nd_save_classifier(K, labels, amr_graphs, opt_lam, opt_ct, ck.is_root_kernel_default, ck.is_inverse_centralize_amr)
        return K, opt_ct, opt_lam
    else:
        if cte.is_parallel_kernel_eval:
            K = pke.eval_kernel_parallel(amr_graphs, amr_graphs, lam=tuned_lambda_fr_protein_state, cosine_threshold=tuned_cs_fr_protein_state)
        else:
            K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=tuned_lambda_fr_protein_state, cosine_threshold=tuned_cs_fr_protein_state, is_sparse=False)
        lrn_nd_save_classifier(K, labels, amr_graphs, tuned_lambda_fr_protein_state, tuned_cs_fr_protein_state, ck.is_root_kernel_default, ck.is_inverse_centralize_amr)


def merge_catalyst_domain_labels(is_catalyst_labels, is_domain_labels):
    labels = is_catalyst_labels + 2*is_domain_labels
    labels = np.mod(labels, 3) #just in case there is a noisy label where same entity is labeled both as domain and catalyst
    return labels


def test_extraction_classifier():
    amr_graphs_train, is_catalyst_labels_train, is_domain_labels_train, _ = get_data(is_train=True)
    num_train = amr_graphs_train.shape[0]
    amr_graphs_test, is_catalyst_labels_test, is_domain_labels_test, _ = get_data(is_train=False)
    num_test = amr_graphs_test.shape[0]
    train = np.arange(num_train)
    test = np.arange(num_test)+num_train
    #merge two arrays
    amr_graphs = np.empty(dtype=np.object, shape=(num_train+num_test, 1))
    amr_graphs[train, 0] = amr_graphs_train[:, 0]
    amr_graphs[test, 0] = amr_graphs_test[:, 0]
    svm_clf = skl_svm.SVC(kernel='precomputed', probability=True, verbose=False, class_weight='auto')
    if not cte.is_multiclass:
        #catalyst
        K_catalyst = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=tuned_lambda_fr_catalyst, cosine_threshold=tuned_cs_fr_catalyst)
        K_catalyst_train = K_catalyst[np.meshgrid(train, train)]
        K_catalyst_test = K_catalyst[np.meshgrid(train, test)]
        print 'Training ...'
        svm_clf.fit(K_catalyst_train, is_catalyst_labels_train)
        print 'Inferring ...'
        catalyst_labels_test_pred = svm_clf.predict(K_catalyst_test)
        print 'Prediction results for catalyst ...'
        precision = skm.precision_score(is_catalyst_labels_test.astype(np.int), catalyst_labels_test_pred.astype(np.int))
        print 'Precision: ', precision
        recall = skm.recall_score(is_catalyst_labels_test.astype(np.int), catalyst_labels_test_pred.astype(np.int))
        print 'Recall: ', recall
        f1 = skm.f1_score(is_catalyst_labels_test.astype(np.int), catalyst_labels_test_pred.astype(np.int))
        print 'F1: ', f1
        auc = skm.roc_auc_score(is_catalyst_labels_test.astype(np.int), catalyst_labels_test_pred.astype(np.int))
        print 'AUC:', auc
        zero_one_loss = skm.zero_one_loss(is_catalyst_labels_test.astype(np.int), catalyst_labels_test_pred.astype(np.int))
        print 'zero_one_loss: ', zero_one_loss
        confusion_m = skm.confusion_matrix(is_catalyst_labels_test.astype(np.int), catalyst_labels_test_pred.astype(np.int))
        print 'confusion matrix: ', confusion_m
        #domain
        K_domain = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=tuned_lambda_fr_domain, cosine_threshold=tuned_cs_fr_domain)
        K_domain_train = K_domain[np.meshgrid(train, train)]
        K_domain_test = K_domain[np.meshgrid(train, test)]
        print 'Training ....'
        svm_clf = skl_svm.SVC(kernel='precomputed', probability=True, verbose=False, class_weight='auto')
        svm_clf.fit(K_domain_train, is_domain_labels_train)
        print 'Inferring ....'
        domain_labels_test_pred = svm_clf.predict(K_domain_test)
        print 'Prediction results for domain ...'
        precision = skm.precision_score(is_domain_labels_test.astype(np.int), domain_labels_test_pred.astype(np.int))
        print 'Precision: ', precision
        recall = skm.recall_score(is_domain_labels_test.astype(np.int), domain_labels_test_pred.astype(np.int))
        print 'Recall: ', recall
        f1 = skm.f1_score(is_domain_labels_test.astype(np.int), domain_labels_test_pred.astype(np.int))
        print 'F1: ', f1
        auc = skm.roc_auc_score(is_domain_labels_test.astype(np.int), domain_labels_test_pred.astype(np.int))
        print 'AUC:', auc
        zero_one_loss = skm.zero_one_loss(is_domain_labels_test.astype(np.int), domain_labels_test_pred.astype(np.int))
        print 'zero_one_loss: ', zero_one_loss
        confusion_m = skm.confusion_matrix(is_domain_labels_test.astype(np.int), domain_labels_test_pred.astype(np.int))
        print 'confusion matrix: ', confusion_m
    else:
        labels_train = merge_catalyst_domain_labels(is_catalyst_labels_train, is_domain_labels_train)
        labels_test = merge_catalyst_domain_labels(is_catalyst_labels_test, is_domain_labels_test)
        K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=tuned_lambda_fr_catalyst_domain, cosine_threshold=tuned_cs_fr_catalyst_domain)
        K_train = K[np.meshgrid(train, train)]
        K_test = K[np.meshgrid(train, test)]
        print 'Training ...'
        svm_clf.fit(K_train, labels_train)
        print 'Inferring ...'
        labels_test_pred = svm_clf.predict(K_test)
        print 'Prediction results for catalyst ...'
        precision = skm.precision_score(labels_test, labels_test_pred)
        print 'Precision: ', precision
        recall = skm.recall_score(labels_test, labels_test_pred)
        print 'Recall: ', recall
        f1 = skm.f1_score(labels_test, labels_test_pred)
        print 'F1: ', f1
        zero_one_loss = skm.zero_one_loss(labels_test, labels_test_pred)
        print 'zero_one_loss: ', zero_one_loss
        confusion_m = skm.confusion_matrix(labels_test, labels_test_pred)
        print 'confusion matrix: ', confusion_m
        lb = skp.LabelBinarizer()
        lb.fit(labels_test)
        auc = skm.roc_auc_score(lb.transform(labels_test), lb.transform(labels_test_pred))
        print 'roc auc : ', auc
        ars = skm.adjusted_rand_score(labels_test, labels_test_pred)
        print 'adjusted random score: ', ars


def test_extraction_classifier_joint(svm_clf=None):
    if not ck.is_inverse_centralize_amr:
        raise NotImplementedError
    amr_graphs_train, labels_train = get_data_joint(is_train=True)
    amr_graphs_test, labels_test = get_data_joint(is_train=False)
    svm_clf = skl_svm.SVC(kernel='precomputed', probability=True, verbose=False, class_weight='auto')
    K_test = gk.eval_graph_kernel_matrix(amr_graphs_train, amr_graphs_test, lam=tuned_lambda_fr_joint, cosine_threshold=tuned_cs_fr_joint)
    K_train = gk.eval_graph_kernel_matrix(amr_graphs_train, amr_graphs_train, lam=tuned_lambda_fr_joint, cosine_threshold=tuned_cs_fr_joint)
    print 'Training ...'
    svm_clf.fit(K_train, labels_train)
    print 'Inferring ...'
    try:
        labels_test_pred = svm_clf.predict(K_test)
    except:
        print 'in catch'
        labels_test_pred = svm_clf.predict(K_test.transpose())
    print 'Prediction results ...'
    precision = skm.precision_score(labels_test, labels_test_pred)
    print 'Precision: ', precision
    recall = skm.recall_score(labels_test, labels_test_pred)
    print 'Recall: ', recall
    f1 = skm.f1_score(labels_test, labels_test_pred)
    print 'F1: ', f1
    zero_one_loss = skm.zero_one_loss(labels_test, labels_test_pred)
    print 'zero_one_loss: ', zero_one_loss
    confusion_m = skm.confusion_matrix(labels_test, labels_test_pred)
    print 'confusion matrix: ', confusion_m
    lb = skp.LabelBinarizer()
    lb.fit(labels_test)
    auc = skm.roc_auc_score(lb.transform(labels_test), lb.transform(labels_test_pred))
    print 'roc auc : ', auc
    ars = skm.adjusted_rand_score(labels_test, labels_test_pred)
    print 'adjusted random score: ', ars
    #
    np.save('labels_test_pred_joint', labels_test_pred)
    #
    #
    np.save('K_train_joint', K_train)
    #
    labels_test_pred_prob = svm_clf.predict_proba(K_test.transpose())
    np.save('labels_test_pred_prob_joint', labels_test_pred_prob)


if __name__ == '__main__':
    import sys
    is_build_classifier = bool(sys.argv[1])
    is_protein_state = bool(sys.argv[2])
    if is_build_classifier:
        if len(sys.argv) > 3:
            is_tuning_clf = bool(sys.argv[3])
            if not is_protein_state:
                build_extraction_classifier(is_tuning_clf=is_tuning_clf)
            else:
                build_protein_state_extraction_classifier(is_tuning_clf=is_tuning_clf)
        else:
            if not is_protein_state:
                build_extraction_classifier()
            else:
                build_protein_state_extraction_classifier()
    else:
        if is_protein_state:
            raise NotImplementedError
        if len(sys.argv) > 3:
            raise AssertionError
        if not cte.is_joint:
            test_extraction_classifier()
        else:
            test_extraction_classifier_joint()

