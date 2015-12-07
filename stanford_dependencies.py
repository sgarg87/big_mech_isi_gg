import json
import re
import constants_absolute_path as cap
import shutil
import extract_from_amr_dot as ead
import pydot as pd
import textwrap as tw
# from config_console_output import *
import preprocess_sentence_fr_dependency_parse as psdp
# import codecs
import difflib as dl
# import copy
# from config import *
import nltk
import codecs
import inspect, os
# import train_extractor as te
import pickle
# import gen_extractor_features_data as gtd


is_alternative = False


def get_const_global_joint_train_dependencies():
    if is_alternative:
        const_global_joint_train_dependencies = './concept_domain_catalyst_joint_train_data_dependencies_ad'
    else:
        const_global_joint_train_dependencies = './concept_domain_catalyst_joint_train_data_dependencies'
    return const_global_joint_train_dependencies


# todo: issues to resolve
# todo: more than one nodes with perfect match
# todo: same node found for concept, catalyst, protein, protein2
# todo: node not found


class StanfordDependencies():
    def __init__(self):
        self.regex_dependency_triplet = re.compile("(?P<edge_label>.*)\((?P<source>.*)\,\ (?P<destination>.*)\)")
        self.regex_org_label = re.compile("(?P<org_label>.*)\-(\d+)")
        self.min_node_label_match_ratio = 0.8
        self.min_sentence_match_ratio = 0.8
        #
        try:
            this_python_file_path = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
            this_python_file_path = this_python_file_path.strip('.py')
            print 'this_python_file_path', this_python_file_path
            self.log = codecs.open('.'+this_python_file_path+'_log.txt', 'w', 'UTF-8')
            # self.log = codecs.open('./log.txt', 'w', 'UTF-8')
        except BaseException as e:
            print e
            self.log = None

    def __del__(self):
        if self.log is not None:
            self.log.close()

    def get_org_label(self, label):
        match = self.regex_org_label.match(label)
        return match.groupdict()['org_label']

    def get_id(self, label):
        org_label = self.get_org_label(label)
        numeric_str = label.strip(org_label).strip('-')
        id = org_label+numeric_str
        return id

    def get_current_sentence_map(self, sentences_map, sentence_frm_dependency):
        max_ratio = 0
        max_sentence_map = None
        max_sentence = None
        for curr_sentence in sentences_map:
            dl_obj = dl.SequenceMatcher(None, sentence_frm_dependency, curr_sentence)
            curr_ratio = dl_obj.quick_ratio()
            max_ratio = max(max_ratio, curr_ratio)
            if curr_ratio == max_ratio:
                max_sentence = curr_sentence
                max_sentence_map = sentences_map[curr_sentence]
        return max_ratio, max_sentence, max_sentence_map

    def build_nodes_tree_frm_dependencies(self, sentence, dependencies_list):
        nodes = {}
        for curr_dependency_map in dependencies_list:
            source = curr_dependency_map['source']
            source_label = self.get_org_label(source)
            source_id = self.get_id(source)
            source = None
            #
            destination = curr_dependency_map['destination']
            destination_label = self.get_org_label(destination)
            destination_id = self.get_id(destination)
            destination = None
            #
            edge_label = curr_dependency_map['edge_label']
            #source not in the nodes_list
            if not nodes.has_key(source_id):
                nodes[source_id] = ead.Node(source_id, source_label)
            if not nodes.has_key(destination_id):
                nodes[destination_id] = ead.Node(destination_id, destination_label)
            if edge_label == 'root':
                assert source_id == 'ROOT0'
                nodes['root'] = nodes[destination_id]
                nodes['root'].is_root = True
            else:
                #linking destination as child of source, there can be multiple children with a given edge_label
                if edge_label not in nodes[source_id].children:
                    nodes[source_id].children[edge_label] = [nodes[destination_id]]
                else:
                    if nodes[destination_id] not in nodes[source_id].children[edge_label]:
                        nodes[source_id].children[edge_label].append(nodes[destination_id])
                #linking source as parent of destination, there can be multiple parents with same edge_label (eg: a protein can be a catalyst for multiple interactions)
                if edge_label not in nodes[destination_id].parents: #if list is empty
                    nodes[destination_id].parents[edge_label] = [nodes[source_id]]
                else:
                    if nodes[source_id] not in nodes[destination_id].parents[edge_label]:
                        nodes[destination_id].parents[edge_label].append(nodes[source_id])
        return nodes, sentence

    def dependencies_to_dot(self, dependencies_list, dot_file_path, sentence=None):
        if sentence is None:
            sentence = 'No text.'
        dot_graph = pd.Dot(sentence, graph_type='digraph', label=tw.fill(sentence, 80))
        pd_nodes_map = {}

        def add_node(id, label):
            node_color = 'white'
            pd_node = pd.Node(id, label=label, style='filled', fillcolor=node_color)
            pd_nodes_map[id] = pd_node
            dot_graph.add_node(pd_node)

        for curr_dependency_map in dependencies_list:
            #add edges
            curr_edge_label = curr_dependency_map['edge_label']
            curr_source = curr_dependency_map['source']
            curr_destination = curr_dependency_map['destination']
            #
            curr_source_label = self.get_org_label(curr_source)
            curr_source_id = self.get_id(curr_source)
            curr_source = None
            #
            curr_destination_label = self.get_org_label(curr_destination)
            curr_destination_id = self.get_id(curr_destination)
            destination = None
            #
            if curr_edge_label == 'root':
                assert curr_source_id == 'ROOT0'
                curr_source_id = curr_destination_id
                curr_source_label = curr_destination_label
            if curr_source_id not in pd_nodes_map:
                add_node(curr_source_id, curr_source_label)
            if curr_destination_id not in pd_nodes_map:
                add_node(curr_destination_id, curr_destination_label)
            curr_pd_edge = pd.Edge(pd_nodes_map[curr_source_id], pd_nodes_map[curr_destination_id], label=curr_edge_label)
            dot_graph.add_edge(curr_pd_edge)
        #
        dot_graph.write(cap.absolute_path+dot_file_path+'.dot')
        dot_graph.write_pdf(cap.absolute_path+dot_file_path+'.pdf')
        return dot_graph

    def get_a_sdg(self, f):
        curr_sentence = f.readline()
        if curr_sentence is None or curr_sentence == '':
            return None
        curr_sentence = psdp.postprocess_sentence_frm_dependency_graph_parse(curr_sentence)
        curr_sentence = "\"" + curr_sentence + "\""
        curr_blank_line_after_sentence = f.readline()
        assert not curr_blank_line_after_sentence.strip(), curr_sentence+'{'+curr_blank_line_after_sentence+'}'
        sdg_dependencies_list = []
        while True:
            curr_dependency = f.readline()
            if not curr_dependency.strip():
                break
            else:
                match = self.regex_dependency_triplet.match(curr_dependency)
                curr_dependency_elements_map = match.groupdict()
                sdg_dependencies_list.append(curr_dependency_elements_map)
        return {'sentence': curr_sentence, 'dependencies': sdg_dependencies_list}

    def load_stanford_dependencies_train_data(self, const_global_joint_train_dependencies=None):
        if const_global_joint_train_dependencies is None:
            const_global_joint_train_dependencies = get_const_global_joint_train_dependencies()
        #
        file_name = const_global_joint_train_dependencies+'.pickle'
        with open(cap.absolute_path+file_name, 'r') as h:
            data = pickle.load(h)
        assert data is not None
        return data

    def load_stanford_dependency_graphs_frm_text_nd_gen_subgraphs(self, file_path, const_global_joint_train_dependencies=None):
        f_txt = codecs.open(file_path+'_sdg.txt', 'r', 'UTF-8')
        f_json = open(file_path+'.json', 'r')
        sentences_map = json.load(f_json)
        # print sentences_map.keys()
        directory_path = file_path+'_dot_files'
        if os.path.exists(cap.absolute_path+directory_path):
            shutil.rmtree(cap.absolute_path+directory_path)
        os.makedirs(cap.absolute_path+directory_path)
        #
        count_subgraphs_not_found_in_dependencies = 0
        count_subgraphs = 0
        #
        data_dependencies = {}
        data_dependencies['paths_map'] = {}
        data_dependencies['interaction_tuples_map'] = {}
        data_dependencies['joint_labels_map'] = {}
        data_dependencies['sentences_map'] = {}
        while True:
            curr_map = self.get_a_sdg(f_txt)
            if curr_map is None:
                break
            #
            curr_max_ratio, curr_max_ratio_sentence, curr_max_ratio_sentence_info_map =\
                self.get_current_sentence_map(sentences_map, curr_map['sentence'])
            if curr_max_ratio < self.min_sentence_match_ratio:
                print 'curr_max_ratio', curr_max_ratio
                raise AssertionError
            else:
                curr_org_amr_dot_file_path = curr_max_ratio_sentence_info_map['org_amr_dot_file_path']
                curr_org_amr_dot_file_path_split = os.path.split(curr_org_amr_dot_file_path)
                curr_org_amr_dot_file_name = curr_org_amr_dot_file_path_split[-1]
                curr_file_path = directory_path + '/' + curr_org_amr_dot_file_name
                self.dependencies_to_dot(curr_map['dependencies'], curr_file_path, sentence=curr_map['sentence'])
                #generate subgraphs here
                for curr_subgraph_path in curr_max_ratio_sentence_info_map:
                    if curr_subgraph_path != 'org_amr_dot_file_path':
                        count_subgraphs += 1
                        #
                        curr_subgraph_sdg_file_path = directory_path + '/' + os.path.split(curr_subgraph_path)[-1]
                        #
                        try:
                            curr_subgraph_interaction_triplet_str_tuple = curr_max_ratio_sentence_info_map[curr_subgraph_path]['str_tuple']
                            curr_label = curr_max_ratio_sentence_info_map[curr_subgraph_path]['label']
                            curr_subgraph_nodes_list, curr_subgraph_interaction_triplet_nodes_tuple =\
                                self.generate_dependency_subgraph(
                                curr_subgraph_interaction_triplet_str_tuple,
                                curr_subgraph_sdg_file_path,
                                curr_map['sentence'],
                                curr_map['dependencies']
                            )
                            assert curr_subgraph_nodes_list is not None and curr_subgraph_nodes_list
                            assert curr_subgraph_interaction_triplet_nodes_tuple is not None and curr_subgraph_interaction_triplet_nodes_tuple
                            data_dependencies['paths_map'][curr_subgraph_sdg_file_path] = curr_subgraph_nodes_list
                            data_dependencies['interaction_tuples_map'][curr_subgraph_sdg_file_path] = curr_subgraph_interaction_triplet_nodes_tuple
                            data_dependencies['joint_labels_map'][curr_subgraph_sdg_file_path] = curr_label
                            data_dependencies['sentences_map'][curr_subgraph_sdg_file_path] = curr_map['sentence']
                        # except LookupError as lookup_error:
                        except BaseException as error:
                            count_subgraphs_not_found_in_dependencies += 1
                            if self.log is not None:
                                self.log.write('\n')
                                self.log.write('\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                                self.log.write('\n could not extract subgraph with following details from the dependencies: ')
                                self.log.write('\n curr_subgraph_path '+ curr_subgraph_path)
                                self.log.write('\n sentence in dependency '+ curr_map['sentence'])
                                self.log.write('\n curr_max_ratio_sentence '+ curr_max_ratio_sentence)
                                self.log.write('\n interaction tuple '+ str(curr_max_ratio_sentence_info_map[curr_subgraph_path]))
                                # self.log.write('\n lookup_error '+ str(lookup_error))
                                self.log.write('\n lookup_error '+ str(error))
                                self.log.write('\n\n')
        if self.log is not None:
            self.log.write('\n count_subgraphs_not_found_in_dependencies '+ str(count_subgraphs_not_found_in_dependencies))
            self.log.write('\n count_subgraphs '+ str(count_subgraphs))
        f_txt.close()
        #
        assert data_dependencies is not None and data_dependencies
        assert data_dependencies['paths_map'] is not None and data_dependencies['paths_map']
        assert data_dependencies['interaction_tuples_map'] is not None and data_dependencies['interaction_tuples_map']
        assert data_dependencies['joint_labels_map'] is not None and data_dependencies['joint_labels_map']
        assert data_dependencies['sentences_map'] is not None and data_dependencies['sentences_map']
        assert len(data_dependencies['paths_map']) == len(data_dependencies['interaction_tuples_map']) \
               == len(data_dependencies['joint_labels_map']) == len(data_dependencies['sentences_map'])
        #
        if const_global_joint_train_dependencies is None:
            const_global_joint_train_dependencies = get_const_global_joint_train_dependencies()
        with open(const_global_joint_train_dependencies+'.pickle', 'wb') as f_dependencies_pickle:
            pickle.dump(data_dependencies, f_dependencies_pickle)

    def load_stanford_dependency_graphs_frm_text(self, sdg_file_path, json_file_path, dot_files_dir_path):
        f_txt = codecs.open(sdg_file_path+'.txt', 'r', 'UTF-8')
        f_json = open(json_file_path+'.json', 'r')
        sentences_map = json.load(f_json)
        if os.path.exists(cap.absolute_path+dot_files_dir_path):
            shutil.rmtree(cap.absolute_path+dot_files_dir_path)
        os.makedirs(cap.absolute_path+dot_files_dir_path)
        #
        while True:
            curr_map = self.get_a_sdg(f_txt)
            if curr_map is None:
                break
            #
            curr_max_ratio, curr_max_ratio_sentence, curr_org_amr_dot_file_path =\
                self.get_current_sentence_map(sentences_map, curr_map['sentence'])
            if curr_max_ratio < self.min_sentence_match_ratio:
                print 'curr_max_ratio', curr_max_ratio
                raise AssertionError, curr_max_ratio_sentence+'{'+curr_map['sentence']+'}'
            else:
                curr_org_amr_dot_file_path_split = os.path.split(curr_org_amr_dot_file_path)
                curr_org_amr_dot_file_name = curr_org_amr_dot_file_path_split[-1]
                curr_file_path = dot_files_dir_path + '/' + curr_org_amr_dot_file_name
                self.dependencies_to_dot(curr_map['dependencies'], curr_file_path, sentence=curr_map['sentence'])
        f_txt.close()

    def match_node_label(self, node_label, label, max_ratio):
        substring_match = self.min_node_label_match_ratio
        is_match = False
        if node_label == label:
            max_ratio = 1
            is_match = True
        else:
            if node_label in label or label in node_label:
                max_ratio = max(max_ratio, substring_match)
                if substring_match == max_ratio:
                    is_match = True
            #
            dl_obj = dl.SequenceMatcher(None, node_label, label)
            curr_ratio = dl_obj.quick_ratio()
            max_ratio = max(max_ratio, curr_ratio)
            if curr_ratio == max_ratio:
                is_match = True
        if is_match:
            return max_ratio

    def find_node(self, nodes_list, org_node_label, sentence=None):
        ls_obj = nltk.stem.LancasterStemmer()
        node_label = ls_obj.stem(org_node_label).upper()
        max_ratio = None
        max_node = None
        # todo: consider case of multiple nodes with same name
        for curr_node in nodes_list:
            curr_node_name = ls_obj.stem(curr_node.name).upper()
            #
            max_ratio1 = self.match_node_label(curr_node_name, node_label, max_ratio)
            #
            max_ratio2 = self.match_node_label(curr_node.name.upper(), node_label, max_ratio)
            #
            max_ratio3 = self.match_node_label(curr_node_name, org_node_label.upper(), max_ratio)
            #
            max_ratio4 = self.match_node_label(curr_node.name.upper(), org_node_label.upper(), max_ratio)
            #
            curr_max_ratio = max(max_ratio1, max_ratio2, max_ratio3, max_ratio4)
            max_ratio = max(curr_max_ratio, max_ratio)
            if max_ratio == curr_max_ratio:
                max_node = curr_node
        if max_ratio is None or max_ratio < self.min_node_label_match_ratio:
            if self.log is not None:
                self.log.write('\n')
                self.log.write('\n *****************************************')
                self.log.write('\n max_ratio '+str(max_ratio))
                self.log.write('\n org_node_label '+ org_node_label)
                self.log.write('\n node_label '+ node_label)
                self.log.write('\n max_node.name '+ max_node.name)
                self.log.write('\n ls_obj.stem(max_node.name).upper() '+ls_obj.stem(max_node.name).upper())
                if sentence is not None:
                    self.log.write('\n'+sentence)
        else:
            return max_node

    def generate_dependency_subgraph(self, interaction_triplet_str_tuple, subgraph_sdg_dot_file_path, sentence, dependencies):
        print '***************************'
        print subgraph_sdg_dot_file_path
        print interaction_triplet_str_tuple
        print sentence
        #
        nodes_map, _ = self.build_nodes_tree_frm_dependencies(sentence, dependencies)
        #
        # interaction concept
        concept_node = self.find_node(nodes_map.values(), interaction_triplet_str_tuple[0], sentence)
        if concept_node is None:
            raise LookupError('concept node not found')
        # catalyst
        if interaction_triplet_str_tuple[1] is None:
            catalyst_node = None
        else:
            catalyst_node = self.find_node(nodes_map.values(), interaction_triplet_str_tuple[1], sentence)
            if catalyst_node is None:
                raise LookupError('catalyst node not found')
            else:
                if catalyst_node == concept_node:
                    raise LookupError('catalyst node found is same as the found concept node')
        #protein node
        protein_node = self.find_node(nodes_map.values(), interaction_triplet_str_tuple[2], sentence)
        if protein_node is None:
            raise LookupError('protein node not found')
        else:
            if protein_node == concept_node:
                raise LookupError('protein node found is same as the found concept node')
        #protein2 node
        if len(interaction_triplet_str_tuple) == 4:
            is_complex_type = True
            protein2_node = self.find_node(nodes_map.values(), interaction_triplet_str_tuple[3], sentence)
            if protein2_node is None:
                raise LookupError('protein 2 node not found')
            else:
                if protein2_node == concept_node:
                    raise LookupError('protein2 node found is same as the found concept node')
        else:
            is_complex_type = False
        #
        if catalyst_node is not None:
            print concept_node
            print catalyst_node
            concept_to_catalyst_path_nodes = ead.search_shortest_undirected_bfs_path(concept_node, catalyst_node)
            assert concept_to_catalyst_path_nodes is not None and concept_to_catalyst_path_nodes, 'path from concept to catalyst not found'
        else:
            concept_to_catalyst_path_nodes = None
        #
        concept_to_protein_path_nodes = ead.search_shortest_undirected_bfs_path(concept_node, protein_node)
        assert concept_to_protein_path_nodes is not None and concept_to_protein_path_nodes, 'path from concept to protein not found'
        #
        if is_complex_type:
            concept_to_protein2_path_nodes = ead.search_shortest_undirected_bfs_path(concept_node, protein2_node)
            assert concept_to_protein2_path_nodes is not None and concept_to_protein2_path_nodes, 'path from concept to protein2 node not found'
        #
        if is_complex_type:
            subgraph_nodes, interaction_nodes_tuple \
                = ead.get_bind_concept_path_nodes(concept_to_catalyst_path_nodes, concept_to_protein_path_nodes,
                                                  concept_to_protein2_path_nodes)
        else:
            subgraph_nodes, interaction_nodes_tuple \
                = ead.get_state_change_concept_path_nodes(concept_to_catalyst_path_nodes, concept_to_protein_path_nodes)
        #
        assert (subgraph_nodes[0].name == concept_node.name and subgraph_nodes[0].id == concept_node.id), \
            'first node in path nodes must be same as concept node'
        # setting root node flag for all nodes
        for curr_path_node in subgraph_nodes:
            curr_path_node.is_root = False
        subgraph_nodes[0].is_root = True #first node is supposed to be concept node
        #generate dot file from nodes list
        ead.nodes_to_dot(subgraph_nodes, subgraph_sdg_dot_file_path, sentence)
        #
        return subgraph_nodes, interaction_nodes_tuple


if __name__ == '__main__':
    import sys
    file_path = sys.argv[1]
    sd_obj = StanfordDependencies()
    sd_obj.load_stanford_dependency_graphs_frm_text_nd_gen_subgraphs(file_path)

