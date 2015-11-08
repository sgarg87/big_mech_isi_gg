import time
import numpy as np
import pickle as p
import graph_kernels as gk
import constants_trained_svm_kernel_file_paths as ctskfp
import extract_from_amr_dot as ead
import train_extractor as te
from config_kernel import *
import config
import constants_absolute_path as cap
import config_hpcc as ch
from config_console_output import *
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import parallel_kernel_eval as pke


def get_list_of_nodes_frm_map(nodes_map):
    nodes_list = [nodes_map['root']]
    for curr_node in nodes_map.values():
        if curr_node not in nodes_list:
            nodes_list.append(curr_node)
    return nodes_list


def get_map_frm_list(nodes_list):
    nodes_map = {}
    for node in nodes_list:
        nodes_map[node.id] = node
    nodes_map['root'] = nodes_list[0]
    return nodes_map


class KernelClassifier:
    # recently, joint and protein state classifiers are switched from SVM to Gaussian processes.
    # file names may still be old (using "SVM" in it)
    def __init__(self, is_joint, is_protein_state=False):
        self.is_joint = is_joint
        self.is_protein_state = is_protein_state
        if not is_protein_state:
            if is_joint:
                file_name = ctskfp.file_name_svm_classifier_multiclass_joint + '.pickle'
            else:
                file_name = ctskfp.file_name_svm_classifier_multiclass_not_joint + '.pickle'
        else:
            file_name = ctskfp.file_name_svm_classifier_protein_state + '.pickle'
        #
        print 'loading the model'
        start_time = time.time()
        with open(cap.absolute_path+file_name, 'r') as h:
            self.trained_clf = p.load(h)
        print 'Time to load the trained svm model (and training samples) was ', time.time()-start_time
        self.svm_clf = self.trained_clf['model']
        assert self.svm_clf == 'gp'
        self.gp_weights = self.trained_clf['gp_weights']
        self.kernel_normalization = self.trained_clf['kernel_normalization']
        self.lmbda = self.trained_clf['parameters']['lambda']
        self.cs = self.trained_clf['parameters']['cs']
        self.gp_c = self.trained_clf['parameters']['gp_c']
        self.gp_bias = self.trained_clf['parameters']['gp_bias']
        #
        if 'is_root_kernel_default' in self.trained_clf['parameters']:
            self.is_root_kernel_default = self.trained_clf['parameters']['is_root_kernel_default']
        else:
            self.is_root_kernel_default = True
        if 'is_inverse_centralize_amr' in self.trained_clf['parameters']:
            self.is_inverse_centralize_amr = self.trained_clf['parameters']['is_inverse_centralize_amr']
        else:
            self.is_inverse_centralize_amr = True
        self.train_amr_graphs = self.trained_clf['train_amrs']
        if 'train_kernel_matrix' in self.trained_clf:
            self.train_kernel_matrix = self.trained_clf['train_kernel_matrix']
        else:
            self.train_kernel_matrix = None
        #
        if 'is_joint_amr_synthetic_edge' in self.trained_clf['parameters']:
            self.is_joint_amr_synthetic_edge = self.trained_clf['parameters']['is_joint_amr_synthetic_edge']
        else:
            self.is_joint_amr_synthetic_edge = None
        #
        if 'is_joint_amr_synthetic_role' in self.trained_clf['parameters']:
            self.is_joint_amr_synthetic_role = self.trained_clf['parameters']['is_joint_amr_synthetic_role']
        else:
            self.is_joint_amr_synthetic_role = None
        #
        if 'is_protein_state_amr_synthetic_edge' in self.trained_clf['parameters']:
            self.is_protein_state_amr_synthetic_edge = self.trained_clf['parameters']['is_protein_state_amr_synthetic_edge']
        else:
            self.is_protein_state_amr_synthetic_edge = None
        #
        if 'is_protein_state_amr_synthetic_role' in self.trained_clf['parameters']:
            self.is_protein_state_amr_synthetic_role = self.trained_clf['parameters']['is_protein_state_amr_synthetic_role']
        else:
            self.is_protein_state_amr_synthetic_role = None
        #
        if 'is_protein_state_subgraph_rooted_at_concept_node' in self.trained_clf['parameters']:
            self.is_protein_state_subgraph_rooted_at_concept_node = self.trained_clf['parameters']['is_protein_state_subgraph_rooted_at_concept_node']
        else:
            self.is_protein_state_subgraph_rooted_at_concept_node = None

    def __str__(self):
        map_fr_print = {}
        map_fr_print['is_joint'] = self.is_joint
        map_fr_print['is_protein_state'] = self.is_protein_state
        map_fr_print['is_root_kernel_default'] = self.is_root_kernel_default
        map_fr_print['is_inverse_centralize_amr'] = self.is_inverse_centralize_amr
        map_fr_print['is_joint_amr_synthetic_edge'] = self.is_joint_amr_synthetic_edge
        map_fr_print['is_joint_amr_synthetic_role'] = self.is_joint_amr_synthetic_role
        map_fr_print['is_protein_state_amr_synthetic_edge'] = self.is_protein_state_amr_synthetic_edge
        map_fr_print['is_protein_state_amr_synthetic_role'] = self.is_protein_state_amr_synthetic_role
        map_fr_print['is_protein_state_subgraph_rooted_at_concept_node'] = self.is_protein_state_subgraph_rooted_at_concept_node
        return str(map_fr_print)

    def infer_frm_svm_saved_classifier(self, test_amr_graph):
        print 'self', self
        print 'test_amr_graph.keys()', test_amr_graph.keys()
        curr_nodes_list = get_list_of_nodes_frm_map(test_amr_graph['nodes'])
        curr_tuple = test_amr_graph['tuple']
        if self.is_joint_amr_synthetic_edge is not None and self.is_joint_amr_synthetic_edge:
            te.add_synthetic_edge_joint_subgraph(curr_nodes_list, curr_tuple)
        if self.is_joint_amr_synthetic_role is not None and self.is_joint_amr_synthetic_role:
            te.set_default_role_fr_all_nodes(curr_nodes_list)
            #
            te.add_synthetic_role_joint_subgraph(curr_nodes_list, curr_tuple)
        #
        if self.is_protein_state_amr_synthetic_edge is not None and (not self.is_protein_state_amr_synthetic_edge):
            te.remove_synthetic_edge_protein_state_sub_graph(curr_nodes_list, curr_tuple)
        if self.is_protein_state_amr_synthetic_role is not None and self.is_protein_state_amr_synthetic_role:
            te.set_default_role_fr_all_nodes(curr_nodes_list)
            #
            te.add_synthetic_role_protein_state_sub_graph(curr_nodes_list, curr_tuple)
        #
        if self.is_inverse_centralize_amr:
            if config.debug:
                print 'performing centralization on ', test_amr_graph['path']
            if self.is_protein_state_subgraph_rooted_at_concept_node is not None and self.is_protein_state_subgraph_rooted_at_concept_node:
                curr_tuple = test_amr_graph['tuple']
                te.change_root_node(curr_nodes_list, curr_tuple[1].id)
                te.eliminate_synthetic_edge_cycles(curr_nodes_list, curr_tuple)
                curr_nodes_list = ead.centralize_amr_at_root_node(curr_nodes_list, curr_tuple[1].id)
            else:
                curr_nodes_list = ead.centralize_amr_at_root_node(curr_nodes_list)
            if (te.is_amr_cyclic(curr_nodes_list) and not is_neighbor_kernel) or (te.is_amr_cyclic_undirected(curr_nodes_list) and is_neighbor_kernel):
                curr_nodes_list = ead.eliminate_first_order_cycles(curr_nodes_list)
        else:
            curr_nodes_list = ead.eliminate_first_order_cycles(curr_nodes_list)
        #
        if not ch.is_hpcc:
            ead.nodes_to_dot(curr_nodes_list, test_amr_graph['path']+'_pi')
        #
        if (not te.is_amr_cyclic(curr_nodes_list) and not is_neighbor_kernel) or (not te.is_amr_cyclic_undirected(curr_nodes_list) and is_neighbor_kernel):
            test_amr_graph['nodes'] = get_map_frm_list(curr_nodes_list)
            test_amr_graph_arr = np.empty(dtype=np.object, shape=(1, 1))
            test_amr_graph_arr[0, 0] = test_amr_graph #assuming only one test sample
            test_amr_graph = test_amr_graph_arr
            test_amr_graph_arr = None
            #
            K_test \
                = gk.eval_graph_kernel_matrix(test_amr_graph,
                                              self.train_amr_graphs,
                                              lam=self.lmbda,
                                              cosine_threshold=self.cs,
                                              is_root_kernel=self.is_root_kernel_default,
                                              is_sparse=False,
                                              is_normalize=False)
            test_norm = gk.graph_kernel_wrapper(nodes1=test_amr_graph[0, 0]['nodes'],
                                          nodes2=test_amr_graph[0, 0]['nodes'],
                                          lam=self.lmbda,
                                          cosine_threshold=self.cs,
                                          is_root_kernel=self.is_root_kernel_default)
            K_test /= np.sqrt(self.kernel_normalization*test_norm)
            #
            score_test_pred = K_test.dot(self.gp_weights) + self.gp_bias
            K_test = None
            test_prob = 1/(1+np.exp(-score_test_pred))
            score_test_pred = None
            assert test_prob.size == 1
            test_prob = test_prob[0]
            print 'test_prob', test_prob
            test_prob_vec = np.zeros(3)
            test_prob_vec[1] = test_prob
            test_prob_vec[0] = 1-test_prob
            test_prob = None
            return test_prob_vec
        else:
            return None

    def test(self):
        if not is_inverse_centralize_amr:
            raise NotImplementedError
        if self.is_protein_state:
            amr_graphs_test, labels_test = te.get_protein_state_data(is_train=False)
        else:
            if not self.is_joint:
                raise NotImplementedError
            amr_graphs_test, labels_test = te.get_data_joint(is_train=False)
        K_train_test = pke.eval_kernel_parallel(
            amr_graphs_test, self.train_amr_graphs, lam=self.lmbda, cosine_threshold=self.cs)
        print 'Inferring ...'
        try:
            labels_test_pred = self.svm_clf.predict(K_train_test)
        except:
            print 'in catch'
            labels_test_pred = self.svm_clf.predict(K_train_test.transpose())
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


if __name__ == '__main__':
    is_protein_state_clf = bool(sys.argv[1])
    if not is_protein_state_clf:
        ie_obj = KernelClassifier(is_joint=True, is_protein_state=False)
    else:
        ie_obj = KernelClassifier(is_joint=None, is_protein_state=True)
    ie_obj.test()

