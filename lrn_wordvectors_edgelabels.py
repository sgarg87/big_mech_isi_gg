import extract_from_amr_dot as ead
import wordvec as wv
import numpy as np
import pickle
import scipy.sparse.linalg as la
import math
import datetime as dt
import constants_absolute_path as cap
from config_console_output import *
import glob
from config import *
import edge_labels as el


debug = True

def get_edge_label_wordvector_file_path(is_amr, algo):
    file_path = './edgelabels_wordvectors'
    if is_amr:
        file_path += '_fr_amrs'
    else:
        file_path += '_fr_dependencies'
    #
    if algo in ['cov', 'ls']:
        file_path += '_algo_' + algo
    else:
        raise AssertionError
    file_path += '.pickle'
    return file_path


class EdgeLabelWordVectors:
    def __init__(self, dot_files_dir_list, is_amr):
        self.is_cov = True
        self.dot_files_dir_list = dot_files_dir_list
        self.is_amr = is_amr
        self.edgelabel_nodes_map = None
        self.nodes_pair_outer_product = None
        self.edge_label_covariance = None
        self.edge_label_wordvectors = None
        self.edge_label_matrices = None
        self.source_vec_matrix_map = None
        self.destination_vec_matrix_map = None

    def get_wordvec_outerprod(self, curr_nodes_pair, nodes_pair_outer_product):
        # order is important for outer product
        word1 = curr_nodes_pair[0]
        word2 = curr_nodes_pair[1]
        if (word1, word2) not in nodes_pair_outer_product and (word2,word1) not in nodes_pair_outer_product:
            word1_vec = wv.get_wordvector(word1)
            word2_vec = wv.get_wordvector(word2)
            #
            if word1_vec is None or word2_vec is None:
                word1_word2_op = None
            else:
                # normalize vectors
                word1_vec /= math.sqrt(np.dot(word1_vec, word1_vec))
                word2_vec /= math.sqrt(np.dot(word2_vec, word2_vec))
                word1_word2_op = np.outer(word1_vec, word2_vec)
            #
            nodes_pair_outer_product[(word1, word2)] = word1_word2_op
        if (word1, word2) in nodes_pair_outer_product:
            return nodes_pair_outer_product[(word1,word2)]
        elif (word2, word1) in nodes_pair_outer_product:
            result_to_return = nodes_pair_outer_product[(word2,word1)]
            if result_to_return is not None:
                return result_to_return.T
            else:
                return None
        else:
            raise AssertionError

    def process_dot_file(self, curr_amr_dot_file):
        print 'processing dot file ', curr_amr_dot_file
        nodes, sentence = ead.build_nodes_tree_from_amr_dot_file(curr_amr_dot_file)
        # this step is very important for accurate learning
        if self.is_amr:
            nodes = ead.simplify_nodes_tree_names(nodes)
            nodes = ead.simplify_nodes_tree_identifiers(nodes)
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
                        curr_tuple = (child.get_name_formatted(), curr_node.get_name_formatted())
                    else:
                        curr_tuple = (curr_node.get_name_formatted(), child.get_name_formatted())
                    if curr_tuple not in self.edgelabel_nodes_map[curr_children_key_inv]:
                        self.edgelabel_nodes_map[curr_children_key_inv].append(curr_tuple)

    def gen_edgelabel_nodes_map(self):
        print 'generating edge label nodes map ...'
        assert self.dot_files_dir_list is not None and self.dot_files_dir_list
        self.edgelabel_nodes_map = {}
        for curr_dot_file_dir in self.dot_files_dir_list:
            print 'curr_dot_file_dir', curr_dot_file_dir
            dot_files_paths_list = glob.glob(curr_dot_file_dir+"*.dot")
            print 'No. of dot files in the directory are ', len(dot_files_paths_list)
            for curr_dot_file_path in dot_files_paths_list:
                self.process_dot_file(curr_dot_file_path)
        if debug:
            print self.edgelabel_nodes_map

    def gen_edge_label_covariance(self):
        print 'generating edge label covariance matrix ...'
        assert self.edgelabel_nodes_map is not None and self.edgelabel_nodes_map
        nodes_pair_outer_product = {}
        self.edge_label_covariance = {}
        for curr_edge_label in self.edgelabel_nodes_map:
            nodes_pair_list = self.edgelabel_nodes_map[curr_edge_label]
            for curr_nodes_pair in nodes_pair_list:
                curr_nodes_pair_outer_prod = self.get_wordvec_outerprod(curr_nodes_pair, nodes_pair_outer_product)
                if curr_nodes_pair_outer_prod is None:
                    continue
                if curr_edge_label not in self.edge_label_covariance:
                    self.edge_label_covariance[curr_edge_label] = curr_nodes_pair_outer_prod
                else:
                    if self.is_cov:
                        self.edge_label_covariance[curr_edge_label] += curr_nodes_pair_outer_prod
                    else:
                        self.edge_label_covariance[curr_edge_label] *= curr_nodes_pair_outer_prod
        if debug:
            print self.edge_label_covariance

    def gen_edge_label_word_vectors_frm_covariance(self):
        print 'generating edge label word vectors ...'
        assert self.edge_label_covariance is not None and self.edge_label_covariance
        self.edge_label_wordvectors = {}
        for curr_edge_label in self.edge_label_covariance:
            if debug:
                print 'curr_edge_label', curr_edge_label
            curr_cov = self.edge_label_covariance[curr_edge_label]
            if curr_cov is None:
                raise AssertionError
            # todo: since covariance is symmetric, word vector for ARG1 and ARG1-of will be same
            # todo: this is not desired
            self.edge_label_wordvectors[curr_edge_label] = self.get_first_eig_vector_of_square_real_matrix(curr_cov, is_symmetric=True)
        print 'learned word vectors for edge labels list below: ', self.edge_label_wordvectors.keys()

    def get_first_eig_vector_of_square_real_matrix(self, A, is_symmetric=False):
        # todo: symmetry information is not used as of now
        n = A.shape[0]
        assert n == A.shape[1] == A.shape[0]
        first_eigen_value, first_eigen_vector = la.eigs(A, k=1, maxiter=n*1000, tol=1e-10)
        if debug:
            print 'first_eigen_value', first_eigen_value
            print 'first_eigen_vector', first_eigen_vector
        dim_eig_vec = A.shape[0]
        return first_eigen_vector.reshape([dim_eig_vec])

    def gen_edge_label_matrices_with_least_squares(self):
        def compute_edge_label_matrix_lst_sq(X, Y):
            print 'X.shape', X.shape
            print 'Y.shape', Y.shape
            A = np.linalg.lstsq(X, Y)[0]
            print 'A', A
            print 'A.shape', A.shape
            assert A is not None
            return A

        print 'generating edge label matrices ...'
        assert self.source_vec_matrix_map is not None and self.source_vec_matrix_map
        assert self.destination_vec_matrix_map is not None and self.destination_vec_matrix_map
        self.edge_label_matrices = {}
        for curr_edge_label in self.source_vec_matrix_map:
            if debug:
                print 'curr_edge_label', curr_edge_label
            assert curr_edge_label in self.destination_vec_matrix_map
            #
            X = self.source_vec_matrix_map[curr_edge_label]
            Y = self.destination_vec_matrix_map[curr_edge_label]
            #
            self.edge_label_matrices[curr_edge_label] = compute_edge_label_matrix_lst_sq(X, Y)
            #
            curr_edge_label_inv = el.get_inverse_of_edge_label(curr_edge_label)
            self.edge_label_matrices[curr_edge_label_inv] = compute_edge_label_matrix_lst_sq(Y, X)
        print 'learned matrices for edge labels list below: ', self.edge_label_matrices.keys()

    def reshape_edge_label_matrices_as_vectors(self):
        self.edge_label_wordvectors = {}
        for curr_edge_label in self.edge_label_matrices:
            curr_matrix = self.edge_label_matrices[curr_edge_label]
            assert curr_matrix.shape[0] == curr_matrix.shape[1] and len(curr_matrix.shape) == 2
            self.edge_label_wordvectors[curr_edge_label] = curr_matrix.flatten()

    def save_word_vectors(self, algo):
        print 'saving edge label word vectors ...'
        assert self.edge_label_wordvectors is not None and self.edge_label_wordvectors
        file_path = get_edge_label_wordvector_file_path(self.is_amr, algo)
        with open(cap.absolute_path+file_path, 'wb') as f_elw:
            pickle.dump(self.edge_label_wordvectors, f_elw)

    def save(self):
        file_path = './lrn_wordvectors_edgelabels_obj_backup.pickle'
        with open(cap.absolute_path+file_path, 'wb') as f_b:
            pickle.dump(self, f_b)

    def lrn_edge_vectors_covariance_method(self):
        print 'started at ', dt.datetime.now().time().isoformat()
        self.gen_edgelabel_nodes_map()
        self.gen_edge_label_covariance()
        self.gen_edge_label_word_vectors_frm_covariance()
        self.save_word_vectors(algo='cov')
        print 'completed at ', dt.datetime.now().time().isoformat()

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

    def lrn_edge_vectors_least_squares_method(self):
        print 'started at ', dt.datetime.now().time().isoformat()
        self.gen_edgelabel_nodes_map()
        self.save()
        self.gen_wordvector_matrices_fr_source_destination_nodes()
        self.gen_edge_label_matrices_with_least_squares()
        self.reshape_edge_label_matrices_as_vectors()
        self.save_word_vectors(algo='ls')
        print 'completed at ', dt.datetime.now().time().isoformat()


if __name__ == '__main__':
    import sys
    is_amr = bool(sys.argv[1])
    algo = sys.argv[2]
    #
    dot_files_dir_list = []
    if is_amr:
        dot_files_dir_list.append('../../all-gold/amrs/dot_files/')
        # dot_files_dir_list.append('../../all-auto/amrs/dot_files/')
    else:
        dot_files_dir_list.append('../../all-gold/dependencies/dot_files/')
    #
    elwv_obj = EdgeLabelWordVectors(dot_files_dir_list=dot_files_dir_list, is_amr=is_amr)
    #
    if algo == 'cov':
        elwv_obj.lrn_edge_vectors_covariance_method()
    elif algo == 'ls':
        elwv_obj.lrn_edge_vectors_least_squares_method()
