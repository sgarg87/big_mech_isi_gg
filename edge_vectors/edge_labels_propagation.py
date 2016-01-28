#
# Written by Sahil Garg (sahilgar@usc.edu, sahil@isi.edu, sahilvk87@gmail.com)
#
# Sahil Garg, Aram Galstyan, Ulf Hermjakob, and Daniel Marcu. Extracting biomolecular interactions using semantic parsing of biomedical text. In Proc. of AAAI, 2016.
#
# Copyright ISI-USC 2015
#
import pickle
import datetime as dt
import glob
import random as r
import numpy as np
import matplotlib.pyplot as plt
import read_dot_file as rdf
import wordvec as wv
import constants_absolute_path as cap
from config import *
import edge_labels as el


debug = False

backup_file_path = './lrn_wordvectors_edgelabels_obj_backup.pickle'


def load_obj():
    with open(cap.absolute_path+backup_file_path, 'rb') as f_b:
        obj = pickle.load(f_b)
    return obj


def get_edge_label_wordvector_file_path(is_amr):
    file_path = './edge_labels_vectors_propagated'
    if is_amr:
        file_path += '_fr_amrs'
    else:
        file_path += '_fr_dependencies'
    #
    file_path += '.pickle'
    return file_path


class EdgeLabelVectorsPropagation:
    def __init__(self, dot_files_dir_list, is_amr):
        self.dot_files_dir_list = dot_files_dir_list
        self.is_amr = is_amr
        self.edgelabel_nodes_map = None
        self.edge_labels_list = None
        self.edge_label_wordvectors = None
        self.edge_label_matrices = None
        self.source_vec_matrix_map = None
        self.destination_vec_matrix_map = None
        self.common_parent_edgelabel_nodes_pair = None
        self.child_i_vec_matrix_map = None
        self.child_j_vec_matrix_map = None
        self.e_p = None
        self.e_l = None
        #
        self.is_lst_sq = True
        self.num_neighbor_samples = 50000
        self.is_normalize = False
        self.is_filter = False
        self.min_num_common_parent_instances = 0
        self.is_curr_edge_label_matrix_in_update = False
        self.is_update = True
        self.is_random_init = False

    def process_dot_file_fr_edgelabel_nodes_map(self, curr_amr_dot_file):
        nodes = self.get_nodes_frm_dot_file(curr_amr_dot_file)
        #iterate over each child of each node to add the tuple in a map with key representing edge label and list of tuples as a value
        for curr_node in nodes.itervalues():
            for curr_children_key in curr_node.children:
                if curr_children_key.lower().endswith('-of'):
                    curr_children_key_inv = el.get_inverse_of_edge_label(curr_children_key)
                    is_inverse_edge = True
                else:
                    curr_children_key_inv = curr_children_key
                    is_inverse_edge = False
                if curr_children_key_inv not in self.edgelabel_nodes_map:
                    self.edgelabel_nodes_map[curr_children_key_inv] = []
                for child in curr_node.children[curr_children_key]:
                    if is_inverse_edge:
                        curr_tuple = (curr_node.get_name_formatted(), child.get_name_formatted())
                    else:
                        curr_tuple = (child.get_name_formatted(), curr_node.get_name_formatted())
                    if curr_tuple not in self.edgelabel_nodes_map[curr_children_key_inv]:
                        self.edgelabel_nodes_map[curr_children_key_inv].append(curr_tuple)

    def get_nodes_frm_dot_file(self, curr_amr_dot_file):
        print 'processing dot file ', curr_amr_dot_file
        nodes, _ = rdf.build_nodes_tree_from_amr_dot_file(curr_amr_dot_file)
        # this step is very important for accurate learning
        if self.is_amr:
            nodes = rdf.simplify_nodes_tree_names(nodes)
            # nodes = rdf.simplify_nodes_tree_identifiers(nodes)
        return nodes

    def process_dot_file_fr_same_parent_edgelabel_children_pairs_map(self, curr_amr_dot_file):
        nodes = self.get_nodes_frm_dot_file(curr_amr_dot_file)
        #iterate over each child of each node to add the tuple in a map with key representing edge label and list of tuples as a value
        for curr_node in nodes.itervalues():
            for curr_edge_label_i in curr_node.children:
                for curr_edge_label_j in curr_node.children:
                    if curr_edge_label_i == curr_edge_label_j:
                        continue
                    elif curr_edge_label_i.lower().endswith('-of'):
                        continue
                    elif curr_edge_label_j.lower().endswith('-of'):
                        continue
                    #
                    curr_common_parent_edge_label_pair = tuple([curr_edge_label_i, curr_edge_label_j])
                    curr_common_parent_edge_label_pair_swap = tuple([curr_edge_label_j, curr_edge_label_i])
                    if curr_common_parent_edge_label_pair not in self.common_parent_edgelabel_nodes_pair \
                            and curr_common_parent_edge_label_pair_swap not in self.common_parent_edgelabel_nodes_pair:
                        self.common_parent_edgelabel_nodes_pair[curr_common_parent_edge_label_pair] = []
                    elif curr_common_parent_edge_label_pair not in self.common_parent_edgelabel_nodes_pair:
                        curr_common_parent_edge_label_pair = curr_common_parent_edge_label_pair_swap
                        curr_common_parent_edge_label_pair_swap = None
                    elif curr_common_parent_edge_label_pair_swap not in self.common_parent_edgelabel_nodes_pair:
                        curr_common_parent_edge_label_pair_swap = None
                    else:
                        raise AssertionError('both ({},{}), and its swap can not be in the map'.format(curr_edge_label_i, curr_edge_label_j))
                    #
                    for curr_child_node_i in curr_node.children[curr_common_parent_edge_label_pair[0]]:
                        for curr_child_node_j in curr_node.children[curr_common_parent_edge_label_pair[1]]:
                            curr_tuple = (curr_child_node_i.get_name_formatted(), curr_child_node_j.get_name_formatted())
                            if curr_tuple not in self.common_parent_edgelabel_nodes_pair[curr_common_parent_edge_label_pair]:
                                self.common_parent_edgelabel_nodes_pair[curr_common_parent_edge_label_pair].append(curr_tuple)

    def gen_edgelabel_nodes_map(self):
        print 'generating edge label nodes map ...'
        assert self.dot_files_dir_list is not None and self.dot_files_dir_list
        self.edgelabel_nodes_map = {}
        for curr_dot_file_dir in self.dot_files_dir_list:
            print 'curr_dot_file_dir', curr_dot_file_dir
            dot_files_paths_list = glob.glob(curr_dot_file_dir+"*.dot")
            print 'No. of dot files in the directory are ', len(dot_files_paths_list)
            for curr_dot_file_path in dot_files_paths_list:
                self.process_dot_file_fr_edgelabel_nodes_map(curr_dot_file_path)
        #
        self.merge_duplicate_case_sensitive_edge_labels_in_edgelabel_nodes_map()
        #
        if debug:
            print self.edgelabel_nodes_map

    def merge_duplicate_case_sensitive_edge_labels_in_edgelabel_nodes_map(self):
        assert self.edgelabel_nodes_map is not None and self.edgelabel_nodes_map
        num_edge_labels = len(self.edgelabel_nodes_map)
        new_edgelabel_nodes_map = {}
        for curr_edge_label in self.edgelabel_nodes_map:
            assert isinstance(self.edgelabel_nodes_map[curr_edge_label], list)
            curr_edge_label_lower_case = curr_edge_label.lower()
            if curr_edge_label_lower_case not in new_edgelabel_nodes_map:
                new_edgelabel_nodes_map[curr_edge_label_lower_case] = self.edgelabel_nodes_map[curr_edge_label]
            else:
                new_edgelabel_nodes_map[curr_edge_label_lower_case] += self.edgelabel_nodes_map[curr_edge_label]
        self.edgelabel_nodes_map = new_edgelabel_nodes_map
        new_edgelabel_nodes_map = None
        assert self.edgelabel_nodes_map is not None and self.edgelabel_nodes_map
        assert len(self.edgelabel_nodes_map) <= num_edge_labels
        print self.edgelabel_nodes_map.keys()

    def merge_duplicate_case_sensitive_edge_labels_in_common_parent_edgelabel_pair_nodes_map(self):
        assert self.common_parent_edgelabel_nodes_pair is not None and self.common_parent_edgelabel_nodes_pair
        num_edge_label_pairs = len(self.common_parent_edgelabel_nodes_pair)
        new_common_parent_edgelabel_nodes_pair = {}
        for curr_edge_label_pair in self.common_parent_edgelabel_nodes_pair:
            assert isinstance(self.common_parent_edgelabel_nodes_pair[curr_edge_label_pair], list)
            assert len(curr_edge_label_pair) == 2
            #
            curr_edge_label_pair_lower_case = []
            curr_edge_label_pair_lower_case.append(curr_edge_label_pair[0].lower())
            curr_edge_label_pair_lower_case.append(curr_edge_label_pair[1].lower())
            curr_edge_label_pair_lower_case = tuple(curr_edge_label_pair_lower_case)
            #
            if curr_edge_label_pair_lower_case not in new_common_parent_edgelabel_nodes_pair:
                new_common_parent_edgelabel_nodes_pair[curr_edge_label_pair_lower_case] \
                    = self.common_parent_edgelabel_nodes_pair[curr_edge_label_pair]
            else:
                new_common_parent_edgelabel_nodes_pair[curr_edge_label_pair_lower_case] \
                    += self.common_parent_edgelabel_nodes_pair[curr_edge_label_pair]
        self.common_parent_edgelabel_nodes_pair = new_common_parent_edgelabel_nodes_pair
        new_common_parent_edgelabel_nodes_pair = None
        assert self.common_parent_edgelabel_nodes_pair is not None and self.common_parent_edgelabel_nodes_pair
        assert len(self.common_parent_edgelabel_nodes_pair) <= num_edge_label_pairs
        print self.common_parent_edgelabel_nodes_pair.keys()

    def set_edge_labels_list(self):
        assert self.source_vec_matrix_map is not None and self.source_vec_matrix_map
        self.edge_labels_list = self.source_vec_matrix_map.keys()
        for curr_edge_label in self.edge_labels_list:
            assert curr_edge_label in self.destination_vec_matrix_map

    def gen_common_parent_edgelabel_pair_nodes_map(self):
        print 'generating common parent edge label pair nodes map ...'
        assert self.dot_files_dir_list is not None and self.dot_files_dir_list
        self.common_parent_edgelabel_nodes_pair = {}
        for curr_dot_file_dir in self.dot_files_dir_list:
            print 'curr_dot_file_dir', curr_dot_file_dir
            dot_files_paths_list = glob.glob(curr_dot_file_dir+"*.dot")
            print 'No. of dot files in the directory are ', len(dot_files_paths_list)
            for curr_dot_file_path in dot_files_paths_list:
                self.process_dot_file_fr_same_parent_edgelabel_children_pairs_map(curr_dot_file_path)
        #
        self.merge_duplicate_case_sensitive_edge_labels_in_common_parent_edgelabel_pair_nodes_map()
        #
        if debug:
            print self.common_parent_edgelabel_nodes_pair

    def compute_sol_to_lin_equation_Y_eq_XA(self, X, Y):
        # Y = X A
        print 'X.shape', X.shape
        print 'Y.shape', Y.shape
        if self.is_lst_sq:
            A = np.linalg.lstsq(X, Y)[0]
        else:
            X_inv = np.linalg.pinv(X)
            A = np.dot(X_inv, Y)
        assert A is not None
        print 'A', A
        print 'A.shape', A.shape
        return A

    def reshape_edge_label_matrices_as_vectors(self):
        self.edge_label_wordvectors = {}
        for curr_edge_label in self.edge_label_matrices:
            curr_matrix = self.edge_label_matrices[curr_edge_label]
            assert curr_matrix.shape[0] == curr_matrix.shape[1] and len(curr_matrix.shape) == 2
            self.edge_label_wordvectors[curr_edge_label] = curr_matrix.flatten()

    def save_edge_vectors(self):
        print 'saving edge label word vectors ...'
        assert self.edge_label_wordvectors is not None and self.edge_label_wordvectors
        file_path = get_edge_label_wordvector_file_path(self.is_amr)
        with open(cap.absolute_path+file_path, 'wb') as f_elw:
            pickle.dump(self.edge_label_wordvectors, f_elw)

    def save(self):
        with open(cap.absolute_path+backup_file_path, 'wb') as f_b:
            pickle.dump(self, f_b)

    def gen_wordvector_matrices_fr_source_destination_nodes(self):
        print 'generating word vector matrices for parent child nodes ...'
        assert self.edgelabel_nodes_map is not None and self.edgelabel_nodes_map
        source_vec_matrix_map = {}
        destination_vec_matrix_map = {}
        for curr_edge_label in self.edgelabel_nodes_map:
            nodes_pair_list = self.edgelabel_nodes_map[curr_edge_label]
            source_vectors_list = []
            destination_vectors_list = []
            for curr_nodes_pair in nodes_pair_list:
                source = curr_nodes_pair[0]
                source_vec = wv.get_wordvector(source)
                source = None
                destination = curr_nodes_pair[1]
                destination_vec = wv.get_wordvector(destination)
                destination = None
                #
                if source_vec is None or destination_vec is None:
                    continue
                #
                source_vectors_list.append(source_vec)
                destination_vectors_list.append(destination_vec)
            assert len(source_vectors_list) == len(destination_vectors_list)
            if source_vectors_list and destination_vectors_list:
                source_vec_matrix_map[curr_edge_label] = np.array(source_vectors_list)
                destination_vec_matrix_map[curr_edge_label] = np.array(destination_vectors_list)
        assert source_vec_matrix_map and destination_vec_matrix_map
        self.source_vec_matrix_map = source_vec_matrix_map
        self.destination_vec_matrix_map = destination_vec_matrix_map

    def gen_wordvector_matrices_fr_common_parent_nodes_pair(self):
        print 'generating word vector matrices for common parent nodes pair ...'
        assert self.common_parent_edgelabel_nodes_pair is not None \
               and self.common_parent_edgelabel_nodes_pair
        child_i_vec_matrix_map = {}
        child_j_vec_matrix_map = {}
        for curr_edge_label_pair in self.common_parent_edgelabel_nodes_pair:
            if curr_edge_label_pair[0] not in self.source_vec_matrix_map:
                assert curr_edge_label_pair[0] not in self.destination_vec_matrix_map
                continue
            elif curr_edge_label_pair[1] not in self.source_vec_matrix_map:
                assert curr_edge_label_pair[1] not in self.destination_vec_matrix_map
                continue
            #
            nodes_pair_list = self.common_parent_edgelabel_nodes_pair[curr_edge_label_pair]
            child_i_vectors_list = []
            child_j_vectors_list = []
            for curr_nodes_pair in nodes_pair_list:
                child_i = curr_nodes_pair[0]
                child_i_vec = wv.get_wordvector(child_i)
                child_i = None
                #
                child_j = curr_nodes_pair[1]
                child_j_vec = wv.get_wordvector(child_j)
                child_j = None
                #
                if child_i_vec is None or child_j_vec is None:
                    continue
                #
                child_i_vectors_list.append(child_i_vec)
                child_j_vectors_list.append(child_j_vec)
            assert len(child_i_vectors_list) == len(child_j_vectors_list)
            if child_i_vectors_list and child_j_vectors_list:
                if self.is_filter and len(child_i_vectors_list) < self.min_num_common_parent_instances:
                    continue
                child_i_vec_matrix_map[curr_edge_label_pair] = np.array(child_i_vectors_list)
                child_j_vec_matrix_map[curr_edge_label_pair] = np.array(child_j_vectors_list)
        assert child_i_vec_matrix_map and child_j_vec_matrix_map
        self.child_i_vec_matrix_map = child_i_vec_matrix_map
        self.child_j_vec_matrix_map = child_j_vec_matrix_map

    def eval_error_in_edge_label_pairs_consistencies(self):
        assert self.child_i_vec_matrix_map is not None and self.child_i_vec_matrix_map
        assert self.child_j_vec_matrix_map is not None and self.child_j_vec_matrix_map
        assert self.edge_label_matrices is not None and self.edge_label_matrices
        #
        error_edge_label_pair_consistency = 0
        for curr_edge_label_pair in self.child_i_vec_matrix_map:
            assert curr_edge_label_pair in self.child_j_vec_matrix_map
            Z_i = self.child_i_vec_matrix_map[curr_edge_label_pair]
            Z_j = self.child_j_vec_matrix_map[curr_edge_label_pair]
            A_i = self.edge_label_matrices[curr_edge_label_pair[0]]
            A_j = self.edge_label_matrices[curr_edge_label_pair[1]]
            Z_i_infer = np.dot(Z_i, A_i)
            Z_j_infer = np.dot(Z_j, A_j)
            Z_diff = Z_i_infer-Z_j_infer
            print 'Z_diff', Z_diff
            curr_error_list = np.linalg.norm(Z_diff, axis=1)
            print 'curr_error_list', curr_error_list
            curr_error = curr_error_list.sum()
            if debug:
                print 'curr_edge_label_pair_error: {}, {}'.format(curr_edge_label_pair, curr_error)
            error_edge_label_pair_consistency += curr_error
        if debug:
            print 'error_edge_label_pair_consistency', error_edge_label_pair_consistency
        #
        self.error_edge_label_pair_consistency = error_edge_label_pair_consistency
        return error_edge_label_pair_consistency

    def eval_error_in_edge_label_local_consistencies(self):
        assert self.source_vec_matrix_map is not None and self.source_vec_matrix_map
        assert self.destination_vec_matrix_map is not None and self.destination_vec_matrix_map
        assert self.edge_label_matrices is not None and self.edge_label_matrices
        #
        error_edge_label_local_consistency = 0
        for curr_edge_label in self.source_vec_matrix_map:
            assert curr_edge_label in self.destination_vec_matrix_map
            X = self.source_vec_matrix_map[curr_edge_label]
            Y = self.destination_vec_matrix_map[curr_edge_label]
            A = self.edge_label_matrices[curr_edge_label]
            Y_infer = np.dot(X, A)
            Y_diff = Y-Y_infer
            print 'Y_diff', Y_diff
            curr_error_list = np.linalg.norm(Y_diff, axis=1)
            print 'curr_error_list', curr_error_list
            curr_error = curr_error_list.sum()
            if debug:
                print 'curr_edge_label_local_error: {}, {}'.format(curr_edge_label, curr_error)
            error_edge_label_local_consistency += curr_error
        if debug:
            print 'error_edge_label_local_consistency', error_edge_label_local_consistency
        #
        self.error_edge_label_local_consistency = error_edge_label_local_consistency
        return error_edge_label_local_consistency

    def generate_edge_label_network_adjacency(self):
        assert self.edge_labels_list is not None and self.edge_labels_list
        n = len(self.edge_labels_list)
        G = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                curr_edge_label_pair = tuple([self.edge_labels_list[i], self.edge_labels_list[j]])
                if curr_edge_label_pair in self.child_i_vec_matrix_map:
                    assert curr_edge_label_pair in self.child_j_vec_matrix_map
                    G[i, j] = 1
                    G[j, i] = 1
        self.G = G
        plt.spy(G)
        plt.savefig('./G.pdf', dpi=300, format='pdf')
        plt.close()

    def normalize_edge_label_matrices(self):
        if not self.is_normalize:
            raise AssertionError, 'normalization not allowed in this algorithm'
        for curr_edge_label in self.edge_label_matrices:
            curr_matrix = self.edge_label_matrices[curr_edge_label]
            curr_matrix_norm = np.linalg.norm(curr_matrix)
            curr_matrix = curr_matrix/float(curr_matrix_norm)
            self.edge_label_matrices[curr_edge_label] = curr_matrix

    def get_power_of_matrix_svd(self, curr_matrix, power_val):
        curr_U, curr_s, curr_V = np.linalg.svd(curr_matrix, full_matrices=False)
        curr_s = np.diag(np.power(curr_s, power_val))
        curr_matrix = np.dot(curr_U, np.dot(curr_s, curr_V))
        return curr_matrix

    def get_power_of_matrix_eig(self, curr_matrix, power_val):
        eig_val, eig_vec = np.linalg.eig(curr_matrix)
        print 'eig_val', eig_val
        eig_val_power = np.diag(np.power(eig_val, power_val))
        eig_vec_inv = np.linalg.inv(eig_vec)
        curr_matrix = np.dot(eig_vec, np.dot(eig_val_power, eig_vec_inv))
        return curr_matrix

    def update_edge_vectors(self):
        assert self.edge_label_matrices is not None and self.edge_label_matrices
        assert self.edge_label_local_consistency_matrices is not None and self.edge_label_local_consistency_matrices
        assert self.edge_label_consistency_matrices is not None and self.edge_label_consistency_matrices
        assert self.G is not None and not np.all(self.G == 0)
        #
        edge_labels_list = np.array(self.edge_labels_list)
        n = len(edge_labels_list)
        if debug:
            print self.G
            print edge_labels_list
        for curr_idx in range(n):
            if debug:
                print '********************'
            curr_edge_label = self.edge_labels_list[curr_idx]
            if debug:
                print 'curr_edge_label', curr_edge_label
            if debug:
                print self.G[curr_idx]
                print np.where(self.G[curr_idx] == 1)
            curr_neighbor_edge_labels = edge_labels_list[np.where(self.G[curr_idx] == 1)]
            if len(curr_neighbor_edge_labels) > self.num_neighbor_samples:
                curr_neighbor_edge_labels = r.sample(curr_neighbor_edge_labels, self.num_neighbor_samples)
            if debug:
                print 'curr_neighbor_edge_labels', curr_neighbor_edge_labels
            #
            if len(curr_neighbor_edge_labels) == 0:
                continue
            #
            Xj = self.source_vec_matrix_map[curr_edge_label]
            Yj = self.destination_vec_matrix_map[curr_edge_label]
            #
            P = np.copy(Xj)
            Q = np.copy(Yj)
            for curr_ng_edge_label in curr_neighbor_edge_labels:
                curr_edge_label_pair = tuple([curr_ng_edge_label, curr_edge_label])
                curr_edge_label_pair_swap = tuple([curr_edge_label, curr_ng_edge_label])
                if curr_edge_label_pair in self.child_i_vec_matrix_map:
                    curr_edge_label_pair_swap = None
                    assert curr_edge_label_pair in self.child_j_vec_matrix_map
                    Wj = self.child_j_vec_matrix_map[curr_edge_label_pair]
                    Wi = self.child_i_vec_matrix_map[curr_edge_label_pair]
                elif curr_edge_label_pair_swap in self.child_i_vec_matrix_map:
                    curr_edge_label_pair = None
                    assert curr_edge_label_pair_swap in self.child_j_vec_matrix_map
                    Wi = self.child_j_vec_matrix_map[curr_edge_label_pair_swap]
                    Wj = self.child_i_vec_matrix_map[curr_edge_label_pair_swap]
                else:
                    raise AssertionError
                Ai = self.edge_label_matrices[curr_ng_edge_label]
                #
                P = np.concatenate((P, Wj))
                Q = np.concatenate((Q, np.dot(Wi, Ai)))
            #
            curr_edge_label_matrix_new = self.compute_sol_to_lin_equation_Y_eq_XA(P, Q)
            #
            if self.is_normalize:
                curr_edge_label_matrix_new /= np.linalg.norm(curr_edge_label_matrix_new)
            #
            assert not np.any(np.isnan(curr_edge_label_matrix_new))
            assert not np.any(np.isinf(curr_edge_label_matrix_new))
            #
            self.edge_label_matrices[curr_edge_label] = curr_edge_label_matrix_new

    def generate_edge_label_pair_consistency_matrices(self):
        assert self.child_i_vec_matrix_map is not None and self.child_i_vec_matrix_map
        assert self.child_j_vec_matrix_map is not None and self.child_j_vec_matrix_map
        self.edge_label_consistency_matrices = {}
        for curr_edge_label_pair in self.child_i_vec_matrix_map:
            assert curr_edge_label_pair in self.child_j_vec_matrix_map
            Z_i = self.child_i_vec_matrix_map[curr_edge_label_pair]
            Z_j = self.child_j_vec_matrix_map[curr_edge_label_pair]
            curr_lst_sq_consistency_matrix = self.compute_sol_to_lin_equation_Y_eq_XA(Z_j, Z_i)
            self.edge_label_consistency_matrices[curr_edge_label_pair] = curr_lst_sq_consistency_matrix
        assert self.edge_label_consistency_matrices

    def generate_edge_label_local_consistency_matrices(self):
        print 'generating edge label matrices ...'
        assert self.source_vec_matrix_map is not None and self.source_vec_matrix_map
        assert self.destination_vec_matrix_map is not None and self.destination_vec_matrix_map
        self.edge_label_local_consistency_matrices = {}
        for curr_edge_label in self.source_vec_matrix_map:
            if debug:
                print 'curr_edge_label', curr_edge_label
            assert curr_edge_label in self.destination_vec_matrix_map
            #
            X = self.source_vec_matrix_map[curr_edge_label]
            Y = self.destination_vec_matrix_map[curr_edge_label]
            #
            self.edge_label_local_consistency_matrices[curr_edge_label] = self.compute_sol_to_lin_equation_Y_eq_XA(X, Y)
        assert self.edge_label_local_consistency_matrices

    def initialize_edge_label_matrices(self):
        print 'generating edge label matrices ...'
        assert self.edge_label_local_consistency_matrices is not None and self.edge_label_local_consistency_matrices
        self.edge_label_matrices = {}
        for curr_edge_label in self.edge_labels_list:
            self.edge_label_matrices[curr_edge_label] \
                = np.copy(self.edge_label_local_consistency_matrices[curr_edge_label])
        assert self.edge_label_matrices
        if self.is_normalize:
            self.normalize_edge_label_matrices()

    def initialize_random_edge_label_matrices(self):
        print 'generating edge label matrices ...'
        assert self.edge_labels_list is not None and self.edge_labels_list
        self.edge_label_matrices = {}
        for curr_edge_label in self.edge_labels_list:
            self.edge_label_matrices[curr_edge_label] \
                = np.random.random((100, 100))
        assert self.edge_label_matrices
        if self.is_normalize:
            self.normalize_edge_label_matrices()

    def lrn_edge_vectors_least_squares_method(self, is_load):
        print 'started at ', dt.datetime.now().time().isoformat()
        #
        if not is_load:
            self.gen_edgelabel_nodes_map()
            self.gen_common_parent_edgelabel_pair_nodes_map()
            self.save()
            self.gen_wordvector_matrices_fr_source_destination_nodes()
            self.save()
            #
            self.set_edge_labels_list()
            self.save()
            #
            self.gen_wordvector_matrices_fr_common_parent_nodes_pair()
            self.save()
        #
        self.generate_edge_label_network_adjacency()
        # self.save()
        #
        self.generate_edge_label_local_consistency_matrices()
        # self.save()
        #
        self.generate_edge_label_pair_consistency_matrices()
        #
        if self.is_random_init:
            e_p = {}
            e_l = {}
            num_trials = 5
            for curr_random_idx in range(num_trials):
                e_p[curr_random_idx] = []
                e_l[curr_random_idx] = []
                self.initialize_random_edge_label_matrices()
                #
                if self.is_update:
                    num_iter = 100
                    for i in range(num_iter):
                        e_p[curr_random_idx].append(self.eval_error_in_edge_label_pairs_consistencies())
                        e_l[curr_random_idx].append(self.eval_error_in_edge_label_local_consistencies())
                        print 'e_p: ', e_p
                        print 'e_l: ', e_l
                        if i < num_iter-1:
                            self.update_edge_vectors()
                else:
                    e_p[curr_random_idx].append(self.eval_error_in_edge_label_pairs_consistencies())
                    e_l[curr_random_idx].append(self.eval_error_in_edge_label_local_consistencies())
                #
                print 'e_p: ', e_p
                print 'e_l: ', e_l
                #
            plt.xlabel('Edge Vector Propagation Iterations')
            plt.ylabel('Log Error (Pairwise)')
            for curr_random_idx in range(num_trials):
                plt.plot(np.log(e_p[curr_random_idx]), 'kx-')
            plt.savefig('./e_p.pdf', format='pdf', dpi=300)
            plt.close()
            #
            plt.xlabel('Edge Vector Propagation Iterations')
            plt.ylabel('Log Error (Local)')
            for curr_random_idx in range(num_trials):
                plt.plot(np.log(e_l[curr_random_idx]), 'kx-')
            plt.savefig('./e_l.pdf', format='pdf', dpi=300)
            plt.close()
        else:
            self.initialize_edge_label_matrices()
            #
            e_p = []
            e_l = []
            if self.is_update:
                num_iter = 100
                for i in range(num_iter):
                    e_p.append(self.eval_error_in_edge_label_pairs_consistencies())
                    e_l.append(self.eval_error_in_edge_label_local_consistencies())
                    print 'e_p: ', e_p
                    print 'e_l: ', e_l
                    if i < num_iter-1:
                        self.update_edge_vectors()
            else:
                e_p.append(self.eval_error_in_edge_label_pairs_consistencies())
                e_l.append(self.eval_error_in_edge_label_local_consistencies())
            #
            print 'e_p: ', e_p
            print 'e_l: ', e_l
            e_p = np.array(e_p)
            e_l = np.array(e_l)
            self.e_p = e_p
            self.e_l = e_l
            #
            np.save('./e_p', e_p)
            np.save('./e_l', e_l)
            #
            plt.plot(np.log(e_p), 'kx-')
            plt.xlabel('Edge Vector Propagation Iterations')
            plt.ylabel('Log Error (Pairwise)')
            plt.savefig('./e_p.pdf', format='pdf', dpi=300)
            plt.close()
            #
            plt.plot(np.log(e_l), 'kx-')
            plt.xlabel('Edge Vector Propagation Iterations')
            plt.ylabel('Log Error (Local)')
            plt.savefig('./e_l.pdf', format='pdf', dpi=300)
            plt.close()
            #
            plt.plot(np.log(e_p+e_l), 'kx-')
            plt.xlabel('Edge Vector Propagation Iterations')
            plt.ylabel('Log Error (Pairwise+Local)')
            plt.savefig('./e_p_sum_e_l.pdf', format='pdf', dpi=300)
            plt.close()
            #
            plt.xlabel('Edge Vector Propagation Iterations')
            plt.ylabel('Log Error')
            plt.plot(np.log(e_p), 'r-', label='Pairwise')
            plt.plot(np.log(e_l), 'b-', label='Local')
            plt.plot(np.log(e_p+e_l), 'k-', label='Pairwise+Local')
            plt.legend()
            plt.savefig('./e_p_e_l_joint.pdf', format='pdf', dpi=300)
            plt.close()
        #
        self.save()
        #
        self.reshape_edge_label_matrices_as_vectors()
        self.save()
        self.save_edge_vectors()
        self.compute_time = dt.datetime.now().time().isoformat()
        print 'completed at ', dt.datetime.now().time().isoformat()
        self.save()


class InverseEdgeVectors:
    def __init__(self, is_amr, is_overwrite=False):
        self.is_amr = is_amr
        self.edge_vectors_map = None
        self.is_overwrite = is_overwrite

    def load_edge_vectors(self):
        print 'loading propagated edge label word vectors ...'
        file_path = get_edge_label_wordvector_file_path(self.is_amr)
        with open(cap.absolute_path+file_path, 'rb') as f_elw:
            edge_vectors_map = pickle.load(f_elw)
        assert edge_vectors_map is not None
        assert self.edge_vectors_map is None
        self.edge_vectors_map = edge_vectors_map

    def add_inverse_edge_vectors(self):
        print 'adding edge vectors for inverse edge labels ...'
        assert self.edge_vectors_map is not None and self.edge_vectors_map
        num_edge_labels = len(self.edge_vectors_map)
        edge_labels_list = self.edge_vectors_map.keys()
        print 'edge_labels_list', edge_labels_list
        for curr_edge_label in edge_labels_list:
            print '**********************************'
            #
            curr_edge_label_inverse = el.get_inverse_of_edge_label(curr_edge_label)
            print '({},{})'.format(curr_edge_label, curr_edge_label_inverse)
            #
            if not self.is_overwrite:
                assert curr_edge_label_inverse not in self.edge_vectors_map, 'inverse edge label already present'
            #
            curr_edge_label_vec = self.edge_vectors_map[curr_edge_label]
            assert len(curr_edge_label_vec.shape) == 1
            assert curr_edge_label_vec.shape[0] == 10000
            #
            curr_edge_label_matrix = curr_edge_label_vec.reshape(100,100)
            curr_edge_label_vec = None
            print 'curr_edge_label_matrix', curr_edge_label_matrix
            #
            curr_edge_label_matrix_inv = np.linalg.pinv(curr_edge_label_matrix)
            curr_edge_label_matrix = None
            print 'curr_edge_label_matrix_inv', curr_edge_label_matrix_inv
            #
            curr_edge_label_vec_inverse = curr_edge_label_matrix_inv.flatten()
            curr_edge_label_matrix_inv = None
            self.edge_vectors_map[curr_edge_label_inverse] = curr_edge_label_vec_inverse
        #
        assert len(self.edge_vectors_map) == 2*num_edge_labels

    def save_edge_vectors(self):
        print 'saving edge vectors ...'
        assert self.edge_vectors_map is not None and self.edge_vectors_map
        file_path = get_edge_label_wordvector_file_path(self.is_amr)
        with open(cap.absolute_path+file_path, 'wb') as f_elw:
            pickle.dump(self.edge_vectors_map, f_elw)

    def process(self):
        self.load_edge_vectors()
        self.add_inverse_edge_vectors()
        self.save_edge_vectors()


if __name__ == '__main__':
    import sys
    is_amr = bool(sys.argv[1])
    #
    dot_files_dir_list = []
    if is_amr:
        dot_files_dir_list.append('./amr_sdg_dot_files/amrs/dot_files/')
    else:
        dot_files_dir_list.append('./amr_sdg_dot_files/dependencies/dot_files/')
    #
    is_load = False
    if is_load:
        elwv_obj = load_obj()
        elwv_obj.edge_label_wordvectors = None
        elwv_obj.edge_label_matrices = None
        elwv_obj.e_p = None
        elwv_obj.e_l = None
    else:
        elwv_obj = EdgeLabelVectorsPropagation(dot_files_dir_list=dot_files_dir_list, is_amr=is_amr)
    #
    elwv_obj.lrn_edge_vectors_least_squares_method(is_load)
    #
    iev_obj = InverseEdgeVectors(is_amr=is_amr, is_overwrite=False)
    iev_obj.process()
