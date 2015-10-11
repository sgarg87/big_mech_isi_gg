import pickle
import textwrap as tw
import pydot as pd
import numpy as np
from interactions import *
import config_extraction as ce
import util as my_util
from wordvec import *
import train_extractor as te
if ce.is_protein_path_lkl_screening:
    import infer_extractor as ie
import config_darpa as dc
import constants_absolute_path as cap
import random as r
import config_hpcc as ch
import biopax_model_obj as bmo
import Levenshtein as l
import node_role_types as nrt
import darpa_participant_types_mapping as dptm
import config_darpa as cd
import protein_state_subgraph_synthetic_edges as pssse
import graph_kernels as gk
from config_console_output import *
import config_groundings as cg
import config_valid_protein as cvp
import edge_labels as el
import difflib as dl


# todo: these parameters were for rule based heuristic extraction, remove these at an appropriate stage
weight_depr = 0.9
depth_limit = 20


mutant_re = re.compile('^[A-Z]\d\d\d[A-Z]$')
site_re = re.compile('^[A-Z]\d\d\d$')
number_only_re = re.compile('^[0-9]+$')


class Node:
    def __init__(self, id, name, children=None, parents=None, is_root=False, weight=1):
        self.id = id
        self.name = no_quotes_regexp.sub('', name)
        self.type = no_quotes_regexp.sub('', name) #after processing of the nodes, the name is updated (say with protein name). Then type will represent if it is a protein, enzyme etc
        #
        # this field is specifically for id (uniprot, chebi) of proteins, locations etc
        self.identifier = None
        self.identifier_prob = None
        #
        if children is None:
            self.children = {}
        else:
            self.children = children
        if parents is None:
            self.parents = {}
        else:
            self.parents = parents
        self.is_root = is_root
        self.state_str = [] #state for a protein, enzyme, protein segment, amino-acid ...
        self.weight = weight
        #this label can be used for some additional information on node which be algorithmic specific (for instance, in kernel evaluation)
        self.dummy_label = None #child
        self.dummy_label_inv = None #parent
        # color of node
        self.color = None
        #real space representation of words
        self.name_wordvec = None
        self.type_wordvec = None
        #this is temporary variable used for keeping track of nodes in path from source up to the self node
        self.temp_path_nodes = None
        # #information on a node as part of path nodes
        # self.is_path_start_node = None
        # self.is_path_end_node = None
        #hidden/private variables
        self.__children_list = None
        self.__parent_list = None
        self.__children_and_inlaws_list = None
        self.__ancestors_list = None
        self.__descendants_undirected_list = None
        self.__descendants_inc_inlaws_list = None
        self.__descendants_list = None

    def reset_parent_children_none(self):
        if debug:
            print 'reseting parent children to none for a node with name', self.name
        self.parents = None
        self.children = None
        self.reset_hidden_none()

    def __str__(self):
        node_str = '\n\n' + self.name + ' (' + self.id + ')' + '[' + self.type + ']'
        if hasattr(self, 'role'):
            node_str += '{' + self.role + '}'
        if self.state_str:
            node_str += str(self.state_str)
        if self.children is not None and self.children:
            node_str += ': children{\n'
            children = self.children
            for child_key in children:
                if not isinstance(children[child_key], list):
                    children_list = [children[child_key]]
                else:
                    children_list = children[child_key]
                for child in children_list:
                    node_str += child_key + ':' + child.name
                    if child.state_str:
                        node_str += str(child.state_str)
                    node_str += ', '
            node_str = node_str.rstrip(' ').rstrip(',')
            node_str += '\n}'
        if self.parents is not None and self.parents:
            node_str += ': parents{\n'
            parents = self.parents
            for parent_key in parents:
                parents_list = parents[parent_key]
                for parent in parents_list:
                    node_str += parent_key + ':' + parent.name
                    if parent.state_str:
                        node_str += str(parent.state_str)
                    node_str += ', '
            node_str = node_str.rstrip(' ').rstrip(',')
            node_str += '\n}'
        return node_str

    def get_name_type(self):
        if self.name is None:
            joint_name = self.Type
        elif self.type is None:
            joint_name = self.name
        else:
            assert (self.name is not None and self.type is not None)
            if self.name == self.type:
                joint_name = self.name
            else:
                joint_name = self.name + ' ' + self.type
        #
        assert (joint_name is not None)
        #
        # concept node
        if not concept_regexp.sub('', joint_name):
            joint_name = concept_num_regexp.sub('', joint_name)
        return joint_name

    def get_name_formatted(self):
        name = self.name
        # concept node case
        if not concept_regexp.sub('', name):
            name = concept_num_regexp.sub('', name)
        return name

    def create_children_list(self):
        if self.__children_list is None:
            children_list = []
            for child_key in self.children.keys():
                child = self.children[child_key]
                if isinstance(child, list): #list of nodes
                    for curr_child in child:
                        # curr_child_idx = child.index(curr_child)
                        # curr_child = copy.copy(curr_child) #todo: verify effects of this change
                        curr_child.dummy_label = child_key
                        # child[curr_child_idx] = curr_child
                    children_list += child
                else:
                    # child = copy.copy(child) #todo: verify effects of this change
                    child.dummy_label = child_key
                    children_list.append(child) #assuming just a single node
            self.__children_list = my_util.unique_list(children_list)
        return self.__children_list

    def create_parent_list(self):
        if self.__parent_list is None:
            parents_list = []
            for parent_key in self.parents.keys():
                parent = self.parents[parent_key]
                if isinstance(parent, list): #list of nodes
                    for curr_parent in parent:
                        # curr_parent_idx = parent.index(curr_parent)
                        # curr_parent = copy.copy(curr_parent)
                        curr_parent.dummy_label_inv = parent_key
                        # parent[curr_parent_idx] = curr_parent
                    parents_list += parent
                else:
                    # parent = copy.copy(parent) #todo: verify effects of this change
                    parents_list.append(parent) #assuming just a single node
            self.__parent_list = my_util.unique_list(parents_list)
        return self.__parent_list

    def create_undirected_children_list(self):
        #unique function should not be required
        return my_util.unique_list(self.create_parent_list()+self.create_children_list())

    def create_children_and_in_laws_list(self):
        if self.__children_and_inlaws_list is None:
            self.__children_and_inlaws_list = get_children_and_child_in_laws(self.create_children_list(), self)
        return self.__children_and_inlaws_list

    def create_ancestors_list(self):
        if self.__ancestors_list is None:
            self.__ancestors_list = get_ancestors(self)
        return self.__ancestors_list

    def create_descendants_inc_inlaws_list(self):
        if self.__descendants_inc_inlaws_list is None:
            self.__descendants_inc_inlaws_list = get_descendants_inc_inlaws(self)
        return self.__descendants_inc_inlaws_list

    def create_descendants_list(self):
        if hasattr(self, '__descendants_list'):
            if self.__descendants_list is None:
                self.__descendants_list = get_descendants(self)
            return self.__descendants_list
        else:
            return get_descendants(self)

    def create_descendants_undirected_list(self):
        if self.__descendants_undirected_list is None:
            self.__descendants_undirected_list = get_descendants_undirected(self)
        return self.__descendants_undirected_list
        # #todo: get this code working through the corresponding private variable
        # descendants = get_descendants_undirected(self)
        # if descendants is None:
        #     raise AssertionError
        # return descendants

    def add_child(self, key, child_node):
        # adding child in map of self
        if self.children is None:
            self.children = {}
        if key in self.children:
            if isinstance(self.children[key], list):
                if child_node in self.children[key]:
                    raise AssertionError
                    return
                else:
                    self.children[key].append(child_node)
            else:
                if child_node == self.children[key]:
                    raise AssertionError
                    return
                else:
                    self.children[key] = [self.children[key]] + [child_node]
        else:
            self.children[key] = child_node
        self.reset_hidden_none()

    def add_parent(self, key, parent_node):
        #adding parent to map of self
        if self.parents is None:
            self.parents = {}
        if key in self.parents:
            if parent_node in self.parents[key]:
                raise AssertionError
                return
            else:
                self.parents[key].append(parent_node)
        else:
            self.parents[key] = [parent_node]
        self.reset_hidden_none()

    def remove_child(self, key, child_node):
        if key in self.children:
            if isinstance(self.children[key], list):
                if child_node in self.children[key]:
                    self.children[key].remove(child_node)
                    if not self.children[key]:
                        self.children.pop(key, None)
                    self.reset_hidden_none()
                else:
                    raise AssertionError
            else:
                if child_node == self.children[key]:
                    self.children.pop(key, None)
                    self.reset_hidden_none()
                else:
                    raise AssertionError
        else:
            raise AssertionError

    def is_child(self, child_node):
        for key in self.children:
            if isinstance(self.children[key], list):
                if child_node in self.children[key]:
                    return True, key
            else:
                if child_node == self.children[key]:
                    return True, key
        return False, None

    def remove_parent(self, key, parent_node):
        if key in self.parents:
            if parent_node in self.parents[key]:
                self.parents[key].remove(parent_node)
                if not self.parents[key]:
                    self.parents.pop(key, None)
                self.reset_hidden_none()
            else:
                raise AssertionError
        else:
            raise AssertionError

    def is_parent(self, parent_node):
        for key in self.parents:
            if parent_node in self.parents[key]:
                return True, key
        return False, None

    def add_parent_child_relationship(self, child, key):
        self.add_child(key, child)
        child.add_parent(key, self)

    def remove_parent_child_relationship(self, child, key):
        self.remove_child(key, child)
        child.remove_parent(key, self)

    def reset_hidden_none(self):
        self.__children_list = None
        self.__parent_list = None
        self.__children_and_inlaws_list = None
        self.__descendants_inc_inlaws_list = None
        self.__descendants_undirected_list = None
        self.__descendants_list = None
        self.__ancestors_list = None
        self.temp_path_nodes = None


def get_descendants_inc_inlaws(root_node, descendants=None):
    if descendants is None:
        descendants = [root_node]
    children = root_node.create_children_and_in_laws_list()
    children_ = list(set(children) - set(descendants))
    descendants += children_
    for child in children_:
        descendants = get_descendants_inc_inlaws(child, descendants)
    descendants = my_util.unique_list(descendants)
    if descendants[0] != root_node:
        descendants.remove(root_node)
        descendants.insert(0, root_node)
    return descendants

def get_descendants(root_node, descendants=None):
    if descendants is None:
        descendants = [root_node]
    children = root_node.create_children_list()
    children_ = list(set(children) - set(descendants))
    descendants += children_
    for child in children_:
        descendants = get_descendants(child, descendants)
    descendants = my_util.unique_list(descendants)
    if descendants[0] != root_node:
        descendants.remove(root_node)
        descendants.insert(0, root_node)
    return descendants

def get_descendants_undirected(root_node, descendants=None):
    if root_node is None:
        raise AssertionError
    if descendants is None:
        descendants = [root_node]
    children = root_node.create_undirected_children_list()
    children_ = list(set(children) - set(descendants))
    descendants += children_
    for child in children_:
        if child is None:
            raise AssertionError
        descendants = get_descendants_undirected(child, descendants)
    descendants = my_util.unique_list(descendants)
    if descendants[0] != root_node:
        descendants.remove(root_node)
        descendants.insert(0, root_node)
    if descendants is None:
        raise AssertionError
    return descendants


def get_ancestors(node, ancestors=None):
    if debug:
        print 'getting ancestors of : ', node.name
    if ancestors is None:
        ancestors = []
    for parent_list in node.parents.values():
        for parent in parent_list:
            if parent not in ancestors:
                ancestors.append(parent)
                #there can be some cycles in AMRs http://www.isi.edu/~ulf/amr/lib/amr-dict.html#cycles
                try:
                    get_ancestors(parent, ancestors)
                except RuntimeError as e:
                    print e.message
                    raise e
    return ancestors


def get_children_and_child_in_laws(children, parent):
    if not is_child_in_law:
        return children
    #basically, we are finding child-in-law of parent node
    child_in_laws = copy.copy(children)
    if debug:
        print 'children in law of parent ' + parent.name + ' are: '
    for child in children:
        grandchildren = child.create_children_list()
        for grandchild in grandchildren:
            for curr_key in grandchild.parents.keys():
                for child_in_law in grandchild.parents[curr_key]:
                    if child_in_law not in children and child_in_law != parent and child_in_law not in parent.create_ancestors_list():#todo:
                        child_in_law.dummy_label = child.dummy_label
                        child_in_laws.insert(child_in_laws.index(child)+1, child_in_law)
                        if debug:
                            print 'child in law added: ', child_in_law.name + '/' + ('' if child_in_law.dummy_label is None else child_in_law.dummy_label)
    child_in_laws = my_util.unique_list(child_in_laws)
    # #that means, some child in laws are found
    # new_child_in_laws = list(set(child_in_laws)-set(children))
    # if new_child_in_laws:
    #     #todo: this is not optimal
    #     return get_children_and_child_in_laws(child_in_laws, parent)
    # else:
    #     return child_in_laws
    return child_in_laws


def build_nodes_tree_from_amr_dot_file(amr_dot_file_path):
    #extract interactions and complex formations (including complex disintegration) from the dot file format for an amr
    amr = pd.graph_from_dot_file(cap.absolute_path+amr_dot_file_path)
    sentence = amr.get_label()
    sentence = sentence.replace('\ ', '')
    nodes = {}
    #iterate over all edges to build a tree of nodes
    edges = amr.get_edge_list()
    # print 'edges: ', edges
    for edge in edges:
        # print 'edge', edge
        # print 'edge: ', edge
        source = edge.get_source()
        # print 'source: ', source
        destination = edge.get_destination()
        # print 'destination: ', destination
        edge_label = edge.get_label()
        edge_label = no_quotes_regexp.sub('', edge_label)
        # print 'edge_label: ', edge_label
        edge = None
        #source not in the nodes_list
        if not nodes.has_key(source):
            source_node = amr.get_node(source)[0]
            nodes[source] = Node(source, source_node.get_label())
            source_node = None
        if not nodes.has_key(destination):
            destination_node = amr.get_node(destination)[0]
            nodes[destination] = Node(destination, destination_node.get_label())
            destination_node = None
        if edge_label == 'TOP':
            if source != destination:
                raise AssertionError
            nodes[root] = nodes[source] #source and destination are same for TOP edge
            nodes[root].is_root = True
        else:
            #linking destination as child of source, there can be multiple children with a given edge_label
            if edge_label not in nodes[source].children:
                nodes[source].children[edge_label] = [nodes[destination]]
            else:
                if nodes[destination] not in nodes[source].children[edge_label]:
                    nodes[source].children[edge_label].append(nodes[destination])
            #linking source as parent of destination, there can be multiple parents with same edge_label (eg: a protein can be a catalyst for multiple interactions)
            if edge_label not in nodes[destination].parents: #if list is empty
                nodes[destination].parents[edge_label] = [nodes[source]]
            else:
                if nodes[source] not in nodes[destination].parents[edge_label]:
                    nodes[destination].parents[edge_label].append(nodes[source])
        # print 'nodes: ', nodes
        # amr.write_pdf(cap.absolute_path+amr_dot_file_path+'.pdf')
    return nodes, sentence


def print_nodes(nodes, file_path):
    nodes_print_status = {}
    for node_key in nodes:
        nodes_print_status[node_key] = False
    f = open(cap.absolute_path+file_path, 'w')
    f.write('#the nodes of tree starting from root node are printed in this file.')
    print_node(nodes[root], f, nodes_print_status)
    f.close()


def print_node(node, f, nodes_print_status, is_recursive=True):
    if not nodes_print_status[node.id]:
        f.write(str(node))
        nodes_print_status[node.id] = True
        if is_recursive:
            for child_list in node.children.values():
                for child in child_list:
                    print_node(child, f, nodes_print_status)
            for parent_list in node.parents.values(): #caution: here parent_list itself is a list since for each key, there can be multiple parents
                for parent in parent_list:
                    if (not parent.parents) and (parent.is_root is False): #if parent do not have parents and parent is not a root node: this is special case where the parent acts as adjective of the node (like phosphorylated protein)
                        print_node(parent, f, nodes_print_status)


def simplify_nodes_tree_names(nodes_map):
    def get_name_str(node):
        list_node_ids_in_subgraph_to_remove = []
        #Here we know that name of this node is 'name'. We need to parse the sub-graph to attain the complete name string.
        #assuming that child of node is only one
        if len(node.children) > 1:
            name_str = ''
            for sub_name_nodes in node.children.values():
                for sub_name_node in sub_name_nodes:
                    if sub_name_node.children:
                        AssertionError('This node should not have any children. If otherwise, change code as per new data')
                    name_str += sub_name_node.name + '-'
                    list_node_ids_in_subgraph_to_remove.append(sub_name_node.id)
            name_str = name_str.rstrip('-')
            return name_str, list_node_ids_in_subgraph_to_remove
        if node.children.has_key('op1'):
            # if len(node.children['op1']) > 1:
            #     raise AssertionError
            name_str = node.children['op1'][0].name
            list_node_ids_in_subgraph_to_remove.append(node.children['op1'][0].id)
            return name_str, list_node_ids_in_subgraph_to_remove
        elif node.children.has_key('and'):
            name_str = ''
            if len(node.children['and']) > 1:
                raise AssertionError
            for sub_name_nodes in node.children['and'][0].children.values():
                for sub_name_node in sub_name_nodes:
                    if sub_name_node.children:
                        AssertionError('This node should not have any children. If otherwise, change code as per new data')
                    name_str += sub_name_node.name + '-'
                    list_node_ids_in_subgraph_to_remove.append(sub_name_node.id)
            name_str = name_str.rstrip('-')
            return name_str, list_node_ids_in_subgraph_to_remove
        else:
            if cd.is_darpa:
                return None, list_node_ids_in_subgraph_to_remove
            else:
                raise AssertionError('unexpected child node. modify code as per the new data ...')

    def remove_list_of_nodes(list_node_ids):
        for curr_id in list_node_ids:
            if curr_id in nodes_map:
                nodes_map.pop(curr_id, None)

    node_keys = copy.deepcopy(nodes_map.keys())
    for node_key in node_keys: #since the dictionary is changing, the node_key may no longer be part of it
        if node_key in nodes_map:
            node = nodes_map[node_key]
            if node.name == 'name':
                if debug:
                    print 'node_key', node_key
                name_str, list_node_ids_in_subgraph_to_remove = get_name_str(node)
                if debug:
                    print 'name_str', name_str
                if 'name' in node.parents:
                    for parent in node.parents['name']: #caution: there can be multiple parents sharing same 'name' node
                        if debug:
                            print 'parent', parent
                        parent.children.pop('name', None)
                        # parent.remove_parent_child_relationship(node, 'name')
                        if name_str is not None:
                            parent.name = name_str #update name of parent
                nodes_map.pop(node_key, None) #the node 'name' can be removed from the nodes_map tree now for the simplification
                remove_list_of_nodes(list_node_ids=list_node_ids_in_subgraph_to_remove)
    return nodes_map


def simplify_nodes_tree_identifiers(nodes_map):
    def get_colon_to_identifier_str(identifier_str):
        def get_identifier_fr_prefix(curr_identifier_prefix, identifier_str):
            m = re.findall(curr_identifier_prefix+'(.+?) ', identifier_str+' ')
            if len(m) > 1:
                raise AssertionError
            if len(m) == 1:
                if not m[0].strip():
                    raise AssertionError
                return curr_identifier_prefix+':'+m[0].strip()

        for curr_identifier_prefix in ['Uniprot', 'CHEBI', 'HGNC', 'PubChem-Compound', 'GO']:
            new_identifier_str = get_identifier_fr_prefix(curr_identifier_prefix, identifier_str)
            if new_identifier_str is not None:
                return new_identifier_str
            new_identifier_str = get_identifier_fr_prefix(curr_identifier_prefix.lower(), identifier_str)
            if new_identifier_str is not None:
                return new_identifier_str
            new_identifier_str = get_identifier_fr_prefix(curr_identifier_prefix.upper(), identifier_str)
            if new_identifier_str is not None:
                return new_identifier_str
        raise NotImplementedError

    def delete_child(curr_node, curr_child_key):
        curr_child_node = curr_node.children[curr_child_key][0]
        curr_node.children.pop(curr_child_key, None)
        nodes_map.pop(curr_child_node.id, None)

    #
    xref = 'xref'
    reg_exp_protein_name_no_special_char = re.compile('[A-Za-z0-9- :/_,;]*')
    node_keys = copy.deepcopy(nodes_map.keys())
    for node_key in node_keys: #since the dictionary is changing, the node_key may no longer be part of it
        if node_key in nodes_map:
            node = nodes_map[node_key]
            if xref in node.children:
                if len(node.children[xref]) != 1:
                    raise AssertionError
                xref_child = node.children[xref][0]
                if 'or' not in xref_child.name.lower():
                    raise AssertionError
                xref_grandchild = xref_child.children['op1'][0]
                #
                prob_str = xref_grandchild.children['prob'][0].name
                prob = float(prob_str)
                #
                print 'node.name', node.name
                if reg_exp_protein_name_no_special_char.sub('', node.name):
                    prob_min_threshold = cg.min_identifier_threshold
                else:
                    prob_min_threshold = cg.min_identifier_threshold
                print 'prob_min_threshold', prob_min_threshold
                #
                if prob > prob_min_threshold:
                    if node.identifier is None or not node.identifier.strip():
                        node.identifier = xref_grandchild.children['value'][0].name
                        node.identifier = get_colon_to_identifier_str(node.identifier)
                        node.identifier_prob = prob
                #
                grandchildren_keys = xref_child.children.keys()
                for curr_grandchild_key in grandchildren_keys:
                    if curr_grandchild_key in xref_child.children:
                        delete_child(xref_child.children[curr_grandchild_key][0], 'prob')
                        delete_child(xref_child.children[curr_grandchild_key][0], 'value')
                        delete_child(xref_child, curr_grandchild_key)
                delete_child(node, xref)
    #
    # assign_identifiers_from_model(nodes_map)
    return nodes_map


def assign_identifiers_from_model(nodes_map):
    def assign_id_to_node(curr_node, id):
        # print 'assigning identifier from model to node', curr_node
        curr_node.identifier = id
        print 'assigned identifier {} from model to {}'.format(curr_node.identifier, curr_node.name)

    for curr_node in nodes_map.values():
        # print 'curr_node', curr_node
        name_str = curr_node.name
        if curr_node.identifier is None or not curr_node.identifier.strip():
            for curr_model_protein_name in bmo.bm_obj.protein_identifier_list_map:
                if bmo.bm_obj.protein_identifier_list_map[curr_model_protein_name] is not None \
                        and bmo.bm_obj.protein_identifier_list_map[curr_model_protein_name]:
                    if name_str == curr_model_protein_name:
                        assign_id_to_node(curr_node, bmo.bm_obj.protein_identifier_list_map[curr_model_protein_name][0])
                        break
        if curr_node.identifier is None or not curr_node.identifier.strip():
            for curr_model_protein_name in bmo.bm_obj.protein_identifier_list_map:
                if bmo.bm_obj.protein_identifier_list_map[curr_model_protein_name] is not None \
                        and bmo.bm_obj.protein_identifier_list_map[curr_model_protein_name]:
                    if name_str.lower() == curr_model_protein_name.lower():
                        assign_id_to_node(curr_node, bmo.bm_obj.protein_identifier_list_map[curr_model_protein_name][0])
                        break
        if curr_node.identifier is None or not curr_node.identifier.strip():
            for curr_model_protein_name in bmo.bm_obj.protein_identifier_list_map:
                if bmo.bm_obj.protein_identifier_list_map[curr_model_protein_name] is not None \
                        and bmo.bm_obj.protein_identifier_list_map[curr_model_protein_name]:
                    if l.ratio(name_str.lower(), curr_model_protein_name.lower()) > 0.85:
                        assign_id_to_node(curr_node, bmo.bm_obj.protein_identifier_list_map[curr_model_protein_name][0])
                        break


def simplify_nodes_tree_mod(nodes):
    node_keys = copy.deepcopy(nodes.keys())
    for node_key in node_keys: #since the dictionary is changing, the node_key may no longer be part of it
        if nodes.has_key(node_key):
            node = nodes[node_key]
            if mod in node.children:
                mod_nodes = node.children['mod']
                for mod_node in mod_nodes:
                    if len(mod_node.children) == 0:
                        mod_type = mod_node.type
                        if mod_type not in (protein_labels+protein_part_labels) and concept_regexp.sub('', mod_type):
                            mod_str = mod_node.name
                            for parent in mod_node.parents['mod']: #caution: there can be multiple parents sharing same 'mod' node
                                #
                                if mod_node in parent.children['mod']:
                                    parent.children['mod'].remove(mod_node)
                                if not parent.children['mod']:
                                    parent.children.pop('mod', None) #remove the child node 'mod' from parent
                                #
                                if mod_str not in [protein, enzyme, all, certain, this]:
                                    if concept_regexp.sub('', parent.name):
                                        parent.name = mod_str + ' ' + parent.name #update name of parent
                                elif mod_str in protein_labels and node.type is None:
                                    node.type = protein
                            nodes.pop(mod_node.id, None) #the node 'mod' can be removed from the nodes tree now for the simplification
            elif node.children.has_key('value') and concept_regexp.sub('', node.name):
                value_node = node.children['value']
                if isinstance(value_node, list):
                    if len(value_node) > 1:
                        raise NotImplementedError
                    else:
                        value_node = value_node[0]
                if len(value_node.children) == 0:
                    value_str = value_node.name
                    for parent in value_node.parents['value']:
                        parent.children.pop('value', None)
                        parent.name = value_str
                    nodes.pop(value_node.id, None)
                # else:
                #     raise NotImplementedError
    return nodes


def simplify_nodes_tree_state(nodes):
    raise DeprecationWarning
    node_keys = copy.deepcopy(nodes.keys())
    for node_key in node_keys: #since the dictionary is changing, the node_key may no longer be part of it
        if nodes.has_key(node_key):
            node = nodes[node_key]
            #if node do not have parents and the node is not a root node, and the node has only one child:
            # this is special case where the node acts as adjective of its child node (like phosphorylated protein)
            if (not node.parents) and (node.is_root is False) and (len(node.children) == 1):
                if node.children.has_key(ARG0) or node.children.has_key(ARG1) or node.children.has_key(ARG2) or node.children.has_key(ARG3):
                    #since only one child, we pick first one
                    (child_key, child), = node.children.items()
                    if not concept_regexp.sub('', node.name): #truncate only if a concept
                        state = alpha_regex.sub('', node.name)
                    else:
                        state = node.name
                    if state not in child.state_str and state in state_labels:
                        child.state_str.append(state)  #name of the parent is state of the child
                    child.parents[child_key].remove(node) #remove parent from the child's parents for the child key (note: there can be multiple parents for same key, we need to eliminate only one parent)
                    node.children.pop(child_key, None) #remove child from the parent's list
                    nodes.pop(node.id, None) #remove the parent node from the nodes list
    return nodes


def extract_interactions(nodes, sentence=None):
    raise DeprecationWarning
    #Here it is assumed that the nodes tree structure is simplified.
    #state change, complex formation, complex disintegration are the interactions (in broader sense) recovered here.
    interactions = {}
    for node in nodes.itervalues():
        node_name = alpha_regex.sub('', node.name)
        if node_name in interaction_labels:
            interaction_list = extract_state_change_interaction(node)
            if (interaction_list is not None) and interaction_list:
                for interaction in interaction_list:
                    interaction.text_sentence = sentence
                    if interactions.has_key('state_change'):
                        interactions['state_change'].append(interaction)
                    else:
                        interactions['state_change'] = [interaction]
        elif node_name in complex_labels:
            interaction_list = extract_complex(node)
            if (interaction_list is not None) and interaction_list:
                for interaction in interaction_list:
                    interaction.text_sentence = sentence
                    if interactions.has_key('complex'):
                        interactions['complex'].append(interaction)
                    else:
                        interactions['complex'] = [interaction]
    return interactions


def operate_on_node(node, depth=0, weight=1, is_protein=True):
    raise DeprecationWarning
    if debug:
        print 'operating on node ', node.name
    if depth < depth_limit:
        depth += 1
        if (node.name == 'or') or (node.name == 'and') or (node.name == macro_molecular_complex):
            nodes = list(node.children.itervalues())
            # if is_protein:
            #     nodes += test_if_protein_else_get_ancestor_protein_node(node, depth, weight*weight_depr)
            # else:
            #     nodes += get_protein_parts(node, depth=depth)
        elif node.children.has_key('example'):
            example_node = node.children['example']
            nodes = list(example_node.children.itervalues())
            # if is_protein:
            #     nodes += test_if_protein_else_get_ancestor_protein_node(node, depth, weight*weight_depr)
            # else:
            #     nodes += get_protein_parts(node, depth=depth)
        else:
            nodes = [node]
        node = None
        new_nodes = []
        for node in nodes:
            if node is not None:
                if is_protein:
                    new_nodes += test_if_protein_else_get_ancestor_protein_node(node, depth, weight*weight_depr)
                else:
                    new_nodes += get_protein_parts(node, depth=depth)
        nodes = None
        nodes = new_nodes
        new_nodes = None
        if nodes:
            return nodes


def extract_state_change_interaction(node):
    raise DeprecationWarning
    interaction_type = alpha_regex.sub('', node.name)
    catalyst_nodes = None
    protein_nodes = None
    if debug:
        print 'interaction_type:', interaction_type
    if interaction_type in interaction_labels:
        if interaction_type in [phosphorylate, dephosphorylate, ubiquitinate]:
            #todo:kinase corresponds to catalyst too
            if node.children.has_key('ARG0'):
                catalyst_nodes = operate_on_node(node.children['ARG0'])
            if node.children.has_key('ARG2') and (catalyst_nodes is None or not catalyst_nodes):
                catalyst_nodes = operate_on_node(node.children['ARG2'])
            if node.children.has_key('condition') and (catalyst_nodes is None or not catalyst_nodes):
                catalyst_nodes = operate_on_node(node.children['condition'])
            if node.children.has_key('ARG1'):
                protein_nodes = operate_on_node(node.children['ARG1'])
            if node.children.has_key('ARG3') and (protein_nodes is None or not protein_nodes):
                protein_nodes = operate_on_node(node.children['ARG3'])
            # if protein_nodes is None or not protein_nodes:
            #     #search protein from position meant for catalysts
            #     if ARG0 in node.children:
            #         protein_nodes = operate_on_node(node.children[ARG0], weight=0.1)
            #     if (protein_nodes is None or not protein_nodes) and ARG2 in node.children:
            #         protein_nodes = operate_on_node(node.children[ARG2], weight=0.1)
            #     if (protein_nodes is None or not protein_nodes) and condition in node.children:
            #         protein_nodes = operate_on_node(node.children[condition], weight=0.1)
            #     if protein_nodes is not None and protein_nodes: #if found protein nodes from catalyst nodes
            #         catalyst_nodes = None
        elif interaction_type == signal:
            if node.children.has_key('ARG1'):
                catalyst_nodes = operate_on_node(node.children['ARG1'])
            if node.children.has_key('ARG2') and (catalyst_nodes is None or not catalyst_nodes):
                catalyst_nodes = operate_on_node(node.children['ARG2'])
            if node.children.has_key('condition') and (catalyst_nodes is None or not catalyst_nodes):
                catalyst_nodes = operate_on_node(node.children['condition'])
            if node.children.has_key(ARG0):
                protein_nodes = operate_on_node(node.children[ARG0])
            # if protein_nodes is None or not protein_nodes:
            #     if ARG1 in node.children:
            #         protein_nodes = operate_on_node(node.children[ARG1], weight=0.1)
            #     if (protein_nodes is None or not protein_nodes) and ARG2 in node.children:
            #         protein_nodes = operate_on_node(node.children[ARG2], weight=0.1)
            #     if (protein_nodes is None or not protein_nodes) and condition in node.children:
            #         protein_nodes = operate_on_node(node.children[condition], weight=0.1)
            #     if protein_nodes is not None and protein_nodes: #if found protein nodes from catalyst nodes
            #         catalyst_nodes = None
        #todo: extract state change interaction corresponding from function-01
        else:
            if node.children.has_key('ARG0'):
                catalyst_nodes = operate_on_node(node.children['ARG0'])
            if node.children.has_key('condition') and (catalyst_nodes is None or not catalyst_nodes):
                catalyst_nodes = operate_on_node(node.children['condition'])
            #assuming only one of the arguments ARG1 or ARG2 is present (for activate, it is ARG1, for phosphorylation related concepts in AMR, it is ARG2. As per discussion with Dr. Ulf Hermjakob)
            if node.children.has_key('ARG1'):
                protein_nodes = operate_on_node(node.children['ARG1'])
            if node.children.has_key('ARG2') and (protein_nodes is None or not protein_nodes):
                protein_nodes = operate_on_node(node.children['ARG2'])
            # if protein_nodes is None or not protein_nodes:
            #     if ARG0 in node.children:
            #         protein_nodes = operate_on_node(node.children[ARG0], weight=0.1)
            #     if (protein_nodes is None or not protein_nodes) and condition in node.children:
            #         protein_nodes = operate_on_node(node.children[condition], weight=0.1)
            #     if (protein_nodes is None or not protein_nodes) and ARG3 in node.children:
            #         protein_nodes = operate_on_node(node.children[ARG3], weight=0.1)
            #     if protein_nodes is not None and protein_nodes: #if found protein nodes from catalyst nodes
            #         catalyst_nodes = None
    else:
        raise NotImplementedError
    is_org_catalyst_nodes = True
    if (catalyst_nodes is None) or (not catalyst_nodes):
        is_org_catalyst_nodes = False
        catalyst_nodes = search_catalyst_nodes(node)
    else:
        catalyst_nodes = {'pos': catalyst_nodes, 'neg': []}
    if debug:
        print 'is_org_catalyst_nodes:', is_org_catalyst_nodes
    interactions = []
    if protein_nodes is not None: #catalyst can be null
        for protein_node in protein_nodes:
            if protein_node is not None:
                curr_weight = protein_node.weight
                new_state_str_list = protein_node.state_str[:]
            else:
                curr_weight = 1
                new_state_str_list = []
            if interaction_type in state_labels:
                new_state_str_list.append(interaction_type)
            #create an interaction without a catalyst
            if (catalyst_nodes is None) or (not catalyst_nodes) or (is_org_catalyst_nodes is False):
                weight_factor = 1
            else:
                weight_factor = 0.5
            #
            interaction = Interaction(protein_node.name, protein_node.state_str, None, None, new_state_str_list, interaction_type, weight=(curr_weight*weight_factor))
            interactions.append(interaction)
            if catalyst_nodes is not None:
                for catalyst_node in catalyst_nodes['pos']:
                    interaction = Interaction(protein_node.name, protein_node.state_str, catalyst_node.name, catalyst_node.state_str, new_state_str_list, interaction_type, is_positive_catalyst=True, weight=(curr_weight*catalyst_node.weight))
                    interactions.append(interaction)
                    catalyst_state_str_list = get_possible_state(catalyst_node)
                    if catalyst_state_str_list is not None and catalyst_state_str_list:
                        if catalyst_node.state_str is not None and catalyst_node.state_str:
                            catalyst_state_str_list = np.concatenate((np.array(catalyst_node.state_str), np.array(catalyst_state_str_list))).tolist()
                        interaction = Interaction(protein_node.name, protein_node.state_str, catalyst_node.name, catalyst_state_str_list, new_state_str_list, interaction_type, is_positive_catalyst=True, weight=(curr_weight*catalyst_node.weight))
                        interactions.append(interaction)
                for catalyst_node in catalyst_nodes['neg']:
                    interaction = Interaction(protein_node.name, protein_node.state_str, catalyst_node.name, catalyst_node.state_str, new_state_str_list, interaction_type, is_positive_catalyst=False, weight=(curr_weight*catalyst_node.weight))
                    interactions.append(interaction)
                    catalyst_state_str_list = get_possible_state(catalyst_node)
                    if catalyst_state_str_list is not None and catalyst_state_str_list:
                        if catalyst_node.state_str is not None and catalyst_node.state_str:
                            catalyst_state_str_list = np.concatenate((np.array(catalyst_state_str_list), np.array(catalyst_node.state_str))).tolist()
                        interaction = Interaction(protein_node.name, protein_node.state_str, catalyst_node.name, catalyst_state_str_list, new_state_str_list, interaction_type, is_positive_catalyst=False, weight=(curr_weight*catalyst_node.weight))
                        interactions.append(interaction)
        return interactions


def get_possible_state(node):
    raise DeprecationWarning
    state_str_list = []
    for parent_list in node.parents.itervalues():
            for parent in parent_list:
                if len(parent.children.keys()) == 1:
                    state = alpha_regex.sub('', parent.name)
                    if state in state_labels and state not in node.state_str:
                        state_str_list.append(state)
    return state_str_list


def search_protein_in(node, protein_choices, weight=1):
    raise DeprecationWarning
    if debug:
        print 'searching for protein in node', node.name
    weight *= weight_depr
    # print 'protein_choices:', protein_choices
    if node.type in (protein_labels + protein_part_labels):
        search_node_result = operate_on_node(node, weight=weight)
        if search_node_result is not None:
            for curr_result_node in search_node_result:
                # print 'weight:', weight
                curr_result_node.weight = weight
            protein_choices += search_node_result
    else:
        for child in node.children.itervalues():
            search_protein_in(child, protein_choices, weight)
    print 'protein_choices:', protein_choices


def search_catalyst_nodes(node, catalyst_choices=None, searched_nodes=None, weight=1, forbidden_node=None):
    raise DeprecationWarning
    if debug:
        print 'searching catalyst in ', node.name
    weight *= weight_depr
    if catalyst_choices is None:
        catalyst_choices = {'pos': [], 'neg': []}
    if searched_nodes is None:
        searched_nodes = []
    if forbidden_node is None:
        forbidden_node = node
    searched_nodes.append(node)
    if node.children.has_key('after'):
        search_protein_in(node.children['after'], catalyst_choices['pos'], weight=weight)
    is_parent = False
    for parent_list in node.parents.itervalues():
        for parent in parent_list:
            is_parent = True
            parent_concept = alpha_regex.sub('', parent.name)
            if debug:
                print 'parent_concept:', parent_concept
            if (parent_concept == 'possible') and parent.children.has_key('domain') and (parent.children['domain'] == node):
                if parent.children.has_key('condition'):
                    condition_node = parent.children['condition']
                    search_protein_in(condition_node, catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'use') and parent.children.has_key('ARG2') and (parent.children['ARG2'] == node):
                if parent.children.has_key('ARG1'):
                    search_protein_in(parent.children['ARG1'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept in ['require', 'need']) and parent.children.has_key('ARG0') and (parent.children['ARG0'] == node):
                if parent.children.has_key('ARG1'):
                    search_protein_in(parent.children['ARG1'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'correlate'):
                if parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                    if parent.children.has_key('ARG2'):
                        search_protein_in(parent.children['ARG2'], catalyst_choices['pos'], weight*0.7)
                if parent.children.has_key('ARG2') and (parent.children['ARG2'] == node):
                    if parent.children.has_key('ARG1'):
                        search_protein_in(parent.children['ARG1'], catalyst_choices['pos'], weight*0.7)
            elif (parent_concept in ['prevent', 'suppress']) and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['neg'], weight*0.8)
                elif parent.children.has_key('ARG3'):
                    search_protein_in(parent.children['ARG3'], catalyst_choices['neg'], weight*0.8)
            elif (parent_concept == 'inhibit') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['neg'], weight*0.8)
            elif (parent_concept == 'impede') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['neg'], weight*0.8)
            elif (parent_concept == 'depend') and parent.children.has_key('ARG0') and (parent.children['ARG0'] == node):
                if parent.children.has_key('ARG1'):
                    catalysts = []
                    search_protein_in(parent.children['ARG1'], catalysts, weight*0.7)
                    catalyst_choices['pos'].extend(catalysts)
                    catalyst_choices['neg'].extend(catalysts)
                    catalysts = None
            elif (parent_concept == 'respond') and parent.children.has_key('ARG0') and (parent.children['ARG0'] == node):
                if parent.children.has_key('ARG1'):
                    search_protein_in(parent.children['ARG1'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'mediate') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'regulate') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'deregulate') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['neg'], weight*0.8)
            elif (parent_concept == 'potentiate') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'diminish') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['neg'], weight*0.8)
            elif (parent_concept == 'enhance') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'condition') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG2'):
                    search_protein_in(parent.children['ARG2'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept in ['cause', 'induce']) and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'affect') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['pos'], weight*0.8)
                if parent.children.has_key('ARG2'):
                    search_protein_in(parent.children['ARG2'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'instigate') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'increase') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'decrease') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['neg'], weight*0.8)
            elif (parent_concept == 'result') and parent.children.has_key('ARG2') and (parent.children['ARG2'] == node):
                if parent.children.has_key('ARG1'):
                    search_protein_in(parent.children['ARG1'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'disrupt') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['neg'], weight*0.8)
            elif (parent_concept == 'facilitate') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG0'):
                    search_protein_in(parent.children['ARG0'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'involve') and parent.children.has_key('ARG1') and (parent.children['ARG1'] == node):
                if parent.children.has_key('ARG2'):
                    search_protein_in(parent.children['ARG2'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == 'need') and parent.children.has_key('purpose') and (parent.children['purpose'] == node):
                if parent.children.has_key('ARG1'):
                    search_protein_in(parent.children['ARG1'], catalyst_choices['pos'], weight*0.8)
            elif (parent_concept == activate) and parent.children.has_key(ARG1) and (parent.children[ARG1] == node):
                if parent.children.has_key(ARG0):
                    search_protein_in(parent.children[ARG0], catalyst_choices['pos'], weight*0.8)
            elif parent_concept == 'lead':
                is_lead = False
                if parent.children.has_key(ARG2) and (parent.children[ARG2] == node):
                    is_lead = True
                elif parent.children.has_key(ARG1) and (parent.children[ARG1] == node):
                    is_lead = True
                if is_lead and parent.children.has_key(ARG0):
                    search_protein_in(parent.children[ARG0], catalyst_choices['pos'], weight*0.8)
            if not catalyst_choices['pos'] and not catalyst_choices['neg']:
                for k in parent.children.keys():
                    if k != node:
                        search_protein_in(parent.children[k], catalyst_choices['pos'], weight*0.2)
                        search_protein_in(parent.children[k], catalyst_choices['neg'], weight*0.2)
    # if not catalyst_choices['pos'] and not catalyst_choices['neg']:
    for parent_list in node.parents.itervalues():
        for parent in parent_list:
            search_catalyst_nodes(parent, catalyst_choices, searched_nodes, weight*0.2, forbidden_node)
    # if not catalyst_choices['pos'] and not catalyst_choices['neg']:
    if not is_parent:
        for child in node.children.itervalues():
            if child != forbidden_node:
                if child not in searched_nodes:
                    search_catalyst_nodes(child, catalyst_choices, searched_nodes, weight*0.1, forbidden_node)
            # for spouse_list in child.parents.itervalues():
            #     for spouse in spouse_list:
            #         # if spouse != node:
            #         if spouse not in searched_nodes:
            #             print 'spouse.name', spouse.name
            #             search_catalyst_nodes(spouse, catalyst_choices, searched_nodes, weight*0.4)
    return catalyst_choices


def get_protein_parts(node, depth=0):
    raise DeprecationWarning
    if debug:
        print 'looking for protein parts from node', node.name
    result_nodes = []
    if depth < depth_limit:
        depth += 1
        if node.type == amino_acid: #in protein_part_labels:
            # print 'node.type:', node.type
            # print 'node.name:', node.name
            result_nodes.append(node)
        # else:
        if not concept_regexp.sub('', node.name): #current node is a concept
            if len(node.children) == 1:
                child = node.children.values()[0]
                result_nodes += get_protein_parts(child, depth=depth)
        if node.parents.has_key('part-of'):
            parent_list = node.parents['part-of']
            num_parents = len(parent_list)
            if num_parents > 1:
                NotImplementedError('Number of parents for "part-of" are expected to be one or zero only. This makes implementation of this function easy and also comply with current AMR representation.')
            elif num_parents == 1:
                result_nodes += get_protein_parts(parent_list[0], depth=depth)
        if node.children.has_key('part'):
            child = node.children['part']
            result_nodes += get_protein_parts(child, depth=depth)
        elif node.children.has_key('mod'):
            child = node.children['mod']
            result_nodes += get_protein_parts(child, depth=depth)
        # if not result_nodes:
        op_result_nodes = operate_on_node(node, depth=depth, is_protein=False)
        if op_result_nodes is not None:
            result_nodes += op_result_nodes
        # if not result_nodes:
        for child in node.children.itervalues():
            result_nodes += get_protein_parts(child, depth=depth)
    return result_nodes


def test_if_protein_else_get_ancestor_protein_node(node, depth=0, weight=1):
    raise DeprecationWarning
    if debug:
        print 'looking for protein from node', node.name
    result_nodes = []
    if depth < depth_limit:
        depth += 1
        if node.type in protein_labels:
            node = copy.copy(node)
            if debug:
                print 'node.type', node.type
                print 'node.name', node.name
            node.weight = weight
            result_nodes.append(node)
            protein_part_nodes = get_protein_parts(node)
            for protein_part_node in protein_part_nodes:
                node.state_str.append(protein_part_node.name)
                node.state_str += protein_part_node.state_str
            for parent_list in node.parents.itervalues():
                if parent_list is not None:
                    for parent in parent_list:
                        if debug:
                            print 'parent.type', parent.type
                            print 'parent.name', parent.name
                        if mutate == alpha_regex.sub('', parent.type):
                            state = mutate
                            if parent.children.has_key(value):
                                state += '-' + parent.children[value].name
                            if state not in node.state_str:
                                node.state_str.append(state)
                            break
                        if not concept_regexp.sub('', parent.type):
                            parent_type_alpha = alpha_regex.sub('', parent.type)
                            if parent_type_alpha in state_labels and parent_type_alpha not in node.state_str:
                                node_copy = copy.deepcopy(node)
                                node_copy.state_str.append(parent_type_alpha)
                                result_nodes.append(node_copy)
            # if node.parents.has_key(part_of):
            #     for parent in node.parents[part_of]:
            #         if parent.type in protein_part_labels:
            #             state = parent.name
            #             if state not in node.state_str and state in state_labels:
            #                 node.state_str.append(state)
            #             node.state_str += parent.state_str
            # if node.children.has_key(part):
            #     child = node.children[part]
            #     if child.type in protein_part_labels:
            #         if child.name not in node.state_str and child.name in state_labels:
            #             node.state_str.append(child.name)
            #         node.state_str += child.state_str
        else:
            if not concept_regexp.sub('', node.name): #current node is a concept
                if len(node.children) == 1:
                    child = node.children.values()[0]
                    result_nodes += test_if_protein_else_get_ancestor_protein_node(child, depth, weight*weight_depr)
                else:
                    for child in node.children.itervalues():
                        result_nodes += test_if_protein_else_get_ancestor_protein_node(child, depth, weight*weight_depr*0.9)
            if node.parents.has_key('part'):
                parent_list = node.parents['part']
                num_parents = len(parent_list)
                if num_parents > 1:
                    NotImplementedError('Number of parents for "part" are expected to be one or zero only. This makes implementation of this function easy and also comply with current AMR representation.')
                elif num_parents == 1:
                    result_nodes += test_if_protein_else_get_ancestor_protein_node(parent_list[0], depth, weight)
            if node.children.has_key('part-of') and (not result_nodes):
                child = node.children['part-of']
                result_nodes += test_if_protein_else_get_ancestor_protein_node(child, depth, weight)
            if node.children.has_key('mod') and (not result_nodes):
                child = node.children['mod']
                result_nodes += test_if_protein_else_get_ancestor_protein_node(child, depth, weight)
            if not result_nodes:
                for parent_list in node.parents.itervalues():
                    for parent in parent_list:
                        result_nodes += test_if_protein_else_get_ancestor_protein_node(parent, depth, weight*weight_depr)
            if not result_nodes:
                op_result_nodes = operate_on_node(node, depth, weight)
                if op_result_nodes is not None:
                    result_nodes += op_result_nodes
    return result_nodes


def extract_complex(node):
    raise DeprecationWarning
    catalyst_nodes = None
    protein1_nodes = []
    protein2_nodes = []
    complex_nodes = []
    interaction_type = alpha_regex.sub('', node.name)
    if interaction_type == 'form':
        if node.children.has_key('ARG0'):
            catalyst_nodes = operate_on_node(node.children['ARG0'])
        if node.children.has_key(condition) and (catalyst_nodes is None or not catalyst_nodes):
            catalyst_nodes = operate_on_node(node.children[condition])
        if node.children.has_key('ARG1'):
            complex_node = node.children['ARG1']
            complex_nodes = operate_on_node(complex_node)
        if node.children.has_key('ARG2'):
            complex_parts_node = node.children['ARG2']
            if alpha_regex.sub('', complex_parts_node.name) == 'and':
                if len(complex_parts_node.children) == 2:
                    protein1_node = complex_parts_node.children['op1']
                    protein1_nodes = operate_on_node(protein1_node)
                    protein2_node = complex_parts_node.children['op2']
                    protein2_nodes = operate_on_node(protein2_node)
                else:
                    raise NotImplementedError('it is assumed that complex formation involves two protein inputs only- not more, not less')
            # else:
            #     raise NotImplementedError
    elif interaction_type in ['bind', 'heterodimerize']:
        if node.children.has_key('ARG3'):
            catalyst_nodes = operate_on_node(node.children['ARG3'])
        if node.children.has_key(condition) and (catalyst_nodes is None or not catalyst_nodes):
            catalyst_nodes = operate_on_node(node.children[condition])
        if node.children.has_key('ARG1'):
            protein1_node = node.children['ARG1']
            protein1_nodes = operate_on_node(protein1_node)
        if node.children.has_key('ARG2'):
            protein2_node = node.children['ARG2']
            protein2_nodes = operate_on_node(protein2_node)
    else:
        raise NotImplementedError
    # if isinstance(protein1_node, list):
    #     if len(protein1_node) == 1:
    #         protein1_node = protein1_node[0]
    #     else:
    #         raise NotImplementedError
    # if isinstance(protein2_node, list):
    #     if len(protein2_node) == 1:
    #         protein2_node = protein2_node[0]
    #     else:
    #         raise NotImplementedError
    #todo: just like, we search a catalyst, we also need to search for a protein binding ith another
    is_org_catalyst_nodes = True
    if (catalyst_nodes is None) or (not catalyst_nodes):
        is_org_catalyst_nodes = False
        catalyst_nodes = search_catalyst_nodes(node)
    else:
        catalyst_nodes = {'pos': catalyst_nodes, 'neg': []}
    complex_forms = []
    if not complex_nodes:
        complex_nodes.append(None)
    if protein1_nodes and protein2_nodes:
        for protein1_node in protein1_nodes:
            for protein2_node in protein2_nodes:
                for complex_node in complex_nodes:
                    # print 'protein1_node.name:', protein1_node.name
                    # print 'protein2_node.name:', protein2_node.name
                    #weight in current loop
                    curr_weight = protein1_node.weight
                    # print '1:', curr_weight
                    curr_weight *= protein2_node.weight
                    # print '2:', curr_weight
                    if complex_node is None:
                        #As of now, AMR do not provide arguments for representing product except in concept form. They should since name of complex can be totally new, and not just combination of names of binding proteins
                        #as per current AMR specification, here, we generate complex name and state explicitly
                        complex_name = protein1_node.name + ':' + protein2_node.name
                        #since complex node is not provided by AMR as of now, we have to make this assumption:
                        # state of two proteins becomes state of the complex
                        complex_state_list = protein1_node.state_str[:]
                        if protein2_node.state_str is not None:
                            new_complex_state_list = np.setdiff1d(np.array(protein2_node.state_str), np.array(complex_state_list))
                            complex_state_list = np.concatenate((complex_state_list, new_complex_state_list)).tolist()
                        # complex_name = None
                        # complex_state_list = []
                    else:
                        curr_weight *= complex_node.weight
                        print 'complex:', curr_weight
                        complex_name = complex_node.name
                        complex_state_list = complex_node.state_str
                    #complex formation without a catalyst
                    if (catalyst_nodes is None) or (not catalyst_nodes) or (is_org_catalyst_nodes is False):
                        weight_factor = 1
                    else:
                        weight_factor = 0.5
                    complex_form = ComplexTypeInteraction(protein1_node.name, protein1_node.state_str, protein2_node.name, protein2_node.state_str, None, None, complex_name, complex_state_list, is_positive_catalyst=True, weight=(curr_weight*weight_factor))
                    complex_forms.append(complex_form)
                    #
                    if catalyst_nodes is not None:
                        if debug:
                            print catalyst_nodes
                        for catalyst_node in catalyst_nodes['pos']:
                            # print 'catalyst_node.weight:', catalyst_node.weight
                            if debug:
                                print catalyst_node
                            catalyst_name = catalyst_node.name
                            catalyst_state_list = catalyst_node.state_str
                            complex_form = ComplexTypeInteraction(protein1_node.name, protein1_node.state_str, protein2_node.name, protein2_node.state_str, catalyst_name, catalyst_state_list, complex_name, complex_state_list, is_positive_catalyst=True, weight=(curr_weight*catalyst_node.weight))
                            complex_forms.append(complex_form)
                            catalyst_state_list_new = get_possible_state(catalyst_node)
                            if catalyst_state_list_new is not None and catalyst_state_list_new:
                                catalyst_state_list = np.concatenate((np.setdiff1d(np.array(catalyst_state_list_new), np.array(catalyst_state_list)), catalyst_state_list))
                                complex_form = ComplexTypeInteraction(protein1_node.name, protein1_node.state_str, protein2_node.name, protein2_node.state_str, catalyst_name, catalyst_state_list, complex_name, complex_state_list, is_positive_catalyst=True, weight=(catalyst_node.weight*curr_weight))
                                complex_forms.append(complex_form)
                        for catalyst_node in catalyst_nodes['neg']:
                            # print 'catalyst_node.weight:', catalyst_node.weight
                            catalyst_name = catalyst_node.name
                            catalyst_state_list = catalyst_node.state_str
                            complex_form = ComplexTypeInteraction(protein1_node.name, protein1_node.state_str, protein2_node.name, protein2_node.state_str, catalyst_name, catalyst_state_list, complex_name, complex_state_list, is_positive_catalyst=False, weight=(catalyst_node.weight*curr_weight))
                            complex_forms.append(complex_form)
                            catalyst_state_list_new = get_possible_state(catalyst_node)
                            if catalyst_state_list_new is not None and catalyst_state_list_new:
                                catalyst_state_list = np.concatenate((np.setdiff1d(np.array(catalyst_state_list_new), np.array(catalyst_state_list)), catalyst_state_list))
                                complex_form = ComplexTypeInteraction(protein1_node.name, protein1_node.state_str, protein2_node.name, protein2_node.state_str, catalyst_name, catalyst_state_list, complex_name, complex_state_list, is_positive_catalyst=False, weight=(catalyst_node.weight*curr_weight))
                                complex_forms.append(complex_form)
        return complex_forms


def print_interactions(interactions, file_path):
    f = open(cap.absolute_path+file_path, 'w')
    #print state change interactions
    if interactions.has_key('state_change'):
        state_change_interactions = interactions['state_change']
        f.write('\n\n#State change interactions follow\n')
        for state_change in state_change_interactions:
            f.write(str(state_change))
            f.write('\n')
            f.write(state_change.english())
        state_change_interactions = None
        state_change = None
    #print complex formation interactions
    if interactions.has_key('complex'):
        complex_formations = interactions['complex']
        f.write('\n\n#Complex formations follow')
        for complex in complex_formations:
            f.write(str(complex))
            f.write('\n')
            f.write(complex.english())
    f.close()


def build_nodes_tree_from_amr_dot_file_and_simplify(amr_dot_file_path):
    nodes, sentence = build_nodes_tree_from_amr_dot_file(amr_dot_file_path)
    nodes = simplify_nodes_tree_names(nodes)
    #sequence is important, mod simplification should come after names simplification
    nodes = simplify_nodes_tree_mod(nodes)
    nodes = simplify_nodes_tree_state(nodes)
    return nodes, sentence


def main(amr_dot_file_path):
    nodes, sentence = build_nodes_tree_from_amr_dot_file(amr_dot_file_path)
    print_nodes(nodes, amr_dot_file_path+'.ntf') #nodes tree format
    nodes = simplify_nodes_tree_names(nodes)
    print_nodes(nodes, amr_dot_file_path+'.ntfs') #nodes tree format simplified
    #sequence is important, mod simplification should come after names simplification
    nodes = simplify_nodes_tree_mod(nodes)
    print_nodes(nodes, amr_dot_file_path+'.ntfs2')
    nodes = simplify_nodes_tree_state(nodes)
    print_nodes(nodes, amr_dot_file_path+'.ntfs3')
    interactions = extract_interactions(nodes, sentence)
    print_interactions(interactions, amr_dot_file_path+'.eif') #extracted interactions format
    save_interactions(amr_dot_file_path, interactions)
    save_sentence(amr_dot_file_path, sentence)
    return nodes, interactions


def save_sentence(amr_dot_file_path, sentence):
    print 'sentence: ', sentence
    f = open(cap.absolute_path+amr_dot_file_path + '_s', 'w')
    f.write(sentence)
    f.write('\n')
    f.close()


def save_interactions(amr_dot_file_path, interactions):
    f = open(cap.absolute_path+amr_dot_file_path + '_ei', 'wb')
    pickle.dump(interactions, f)
    f.close()


def load_interactions(amr_dot_file_path):
    f = open(cap.absolute_path+amr_dot_file_path + '_ei', 'rb')
    interactions = pickle.load(f)
    f.close()
    return interactions


def parse_interactions(comma_sep_file_path):
    interactions = {}
    with open(cap.absolute_path+comma_sep_file_path, 'r') as f:
        for curr_line in f:
            if debug:
                print curr_line
            curr_line = curr_line.replace('\t', '').replace('\r', '').replace('\n', '').strip()
            if curr_line:
                interaction_elem = curr_line.split(',')
                if debug:
                    print 'interaction_elem:', interaction_elem
                if len(interaction_elem) == 6:
                    if debug:
                        print 'its a state change interaction'
                    if interaction_elem[0]:
                        catalyst_name = interaction_elem[0]
                    else:
                        catalyst_name = None
                    if interaction_elem[1]:
                        catalyst_state = interaction_elem[1].split(';')
                    else:
                        catalyst_state = []
                    if interaction_elem[2]:
                        protein_name = interaction_elem[2]
                    else:
                        protein_name = None
                    if interaction_elem[3]:
                        protein_state = interaction_elem[3].split(';')
                    else:
                        protein_state = []
                    if interaction_elem[4]:
                        protein_state_new = interaction_elem[4].split(';')
                    else:
                        protein_state_new = []
                    is_positive_catalyst = True
                    if interaction_elem[5]:
                        is_positive_catalyst = bool(int(interaction_elem[5]))
                    interaction = Interaction(protein_name, protein_state, catalyst_name, catalyst_state, protein_state_new, None, is_positive_catalyst, 1)
                    if interactions.has_key(state_change):
                        interactions[state_change].append(interaction)
                    else:
                        interactions[state_change] = [interaction]
                elif len(interaction_elem) == 9:
                    if debug:
                        print 'its a complex formation'
                    if interaction_elem[0]:
                        catalyst_name = interaction_elem[0]
                    else:
                        catalyst_name = None
                    if interaction_elem[1]:
                        catalyst_state = interaction_elem[1].split(';')
                    else:
                        catalyst_state = []
                    if interaction_elem[2]:
                        protein1_name = interaction_elem[2]
                    else:
                        protein1_name = None
                    if interaction_elem[3]:
                        protein1_state = interaction_elem[3].split(';')
                    else:
                        protein1_state = []
                    if interaction_elem[4]:
                        protein2_name = interaction_elem[4]
                    else:
                        protein2_name = None
                    if interaction_elem[5]:
                        protein2_state = interaction_elem[5].split(';')
                    else:
                        protein2_state = []
                    if interaction_elem[6]:
                        complex_name = interaction_elem[6]
                    else:
                        complex_name = None
                    if interaction_elem[7]:
                        complex_state = interaction_elem[7].split(';')
                    else:
                        complex_state = []
                    is_positive_catalyst = True
                    if interaction_elem[8]:
                        is_positive_catalyst = bool(int(interaction_elem[8]))
                    complex_form = ComplexTypeInteraction(protein1_name, protein1_state, protein2_name, protein2_state, catalyst_name, catalyst_state, complex_name, complex_state, is_positive_catalyst, 1)
                    if interactions.has_key(complex):
                        interactions[complex].append(complex_form)
                    else:
                        interactions[complex] = [complex_form]
                else:
                    raise AssertionError
    if debug:
        print interactions
    return interactions


def nodes_to_dot(nodes, dot_file_path, sentence=None):
    if sentence is None:
        sentence = 'No text.'
    dot_graph = pd.Dot(sentence, graph_type='digraph', label=tw.fill(sentence, 80))
    pd_nodes_map = {}

    def add_node(node, is_redundant=False):
        if is_redundant:
            pass
            # raise AssertionError
        if (not hasattr(node, 'color')) or node.color is None:
            if is_redundant:
                node_color = 'red'
            else:
                node_color = 'white'
        else:
            node_color = node.color
        if node.id not in pd_nodes_map:
            label = node.name
            if node.name != node.type:
                label += ' ' + node.type
            if hasattr(node, 'identifier') and node.identifier is not None and node.identifier:
                label += '[' + node.identifier + ']'
            if node.is_root:
                label += ' (root)'
            pd_node = pd.Node(node.id, label=label, style='filled', fillcolor=node_color)
            pd_nodes_map[node.id] = pd_node
            dot_graph.add_node(pd_node)

    edges_key_list = []
    if isinstance(nodes, list):
        nodes_list = nodes
    elif isinstance(nodes, dict):
        nodes_list = nodes.values()
    for node in nodes_list:
        #add nodes
        add_node(node)
    for node in nodes_list:
        #add edges
        for edge_label in node.children:
            children = node.children[edge_label]
            if not isinstance(children, list):
                children = [children]
            for child in children:
                if child.id not in pd_nodes_map:
                    add_node(child, is_redundant=True)
                key = (node.id, edge_label, child.id)
                if key not in edges_key_list:
                    add_node(child) #todo: not required if child in nodes list are considered only
                    edges_key_list.append(key)
                    pd_edge = pd.Edge(pd_nodes_map[node.id], pd_nodes_map[child.id], label=edge_label)
                    dot_graph.add_edge(pd_edge)
                # else:
                #     raise AssertionError
    #todo: write file
    if not ch.is_hpcc and ce.save_subgraph_dot:
        dot_graph.write(cap.absolute_path+dot_file_path+'.dot')
        dot_graph.write_pdf(cap.absolute_path+dot_file_path+'.pdf')
    return dot_graph


def search_shortest_undirected_bfs_path(source, target, queue=None, old_queue=None):
    # todo: verify this code
    # todo: the copy, deep copy may be giving weird results
    #BFS is good enough as long as we have equal weight for all edges
    if source == target:
        raise AssertionError('Source and target refer to same node')
    if source is None or target is None:
        raise AssertionError('Source and target nodes must not be None.')
    source.temp_path_nodes = []
    if queue is None:
        queue = []
        queue.append(source)
        if old_queue is not None:
            raise AssertionError
        old_queue = []
    if not queue:
        return None
    # get the next node from the queue
    curr_node = queue[0]
    queue.remove(curr_node)
    old_queue.append(curr_node) #keep track of all nodes traversed
    if debug:
        print 'picked from queue ', curr_node.name
    if curr_node == target:
        path_nodes = copy.deepcopy(curr_node.temp_path_nodes + [target])
        if path_nodes[0].name != source.name or path_nodes[0].id != source.id:
            raise AssertionError
        elif path_nodes[-1].name != target.name or path_nodes[-1].id != target.id:
            raise AssertionError
        if debug:
            print 'optimized path nodes are ',
            for curr_path_node in path_nodes:
                print curr_path_node
        return path_nodes
    undirected_children = curr_node.create_undirected_children_list()
    for undirected_child in undirected_children:
        if undirected_child not in queue and undirected_child not in old_queue: #avoiding loops
            new_path = copy.copy(curr_node.temp_path_nodes)
            new_path.append(curr_node)
            undirected_child.temp_path_nodes = new_path
            queue.append(undirected_child)
    return search_shortest_undirected_bfs_path(source, target, queue, old_queue)


def assert_name_id_match_fr_nodes(node1, node2):
    if node1.name != node2.name or node1.id != node2.id:
        raise AssertionError


def filter_protein_nodes_of_model(protein_nodes_list, protein_name_idlist_map):
    if protein_nodes_list is None or not protein_nodes_list:
        return protein_nodes_list
    new_protein_nodes_list = []
    for curr_protein_node in protein_nodes_list:
        if curr_protein_node is None:
            continue
        is_added = False
        if curr_protein_node.identifier is not None and curr_protein_node.identifier.strip():
            if len(curr_protein_node.identifier.split(',')) > 1:
                raise NotImplementedError
            if curr_protein_node.identifier in bmo.bm_obj.identifiers_list:
                new_protein_nodes_list.append(curr_protein_node)
                is_added = True
        if not cd.is_entity_in_model_by_identifier_only:
            if not is_added:
                for curr_protein_name_model in bmo.bm_obj.protein_identifier_list_map.keys():
                    if curr_protein_node.name.lower() in curr_protein_name_model.lower():
                        new_protein_nodes_list.append(curr_protein_node)
                        is_added = True
                    elif curr_protein_name_model.lower() in curr_protein_node.name.lower():
                        new_protein_nodes_list.append(curr_protein_node)
                        is_added = True
    return new_protein_nodes_list


def match_protein_name(protein, protein_gold, min_ratio=0.75):
    is_match = False
    if protein == protein_gold:
        is_match = True
    elif protein.lower() == protein_gold.lower():
        is_match = True
    else:
        if protein in protein_gold or protein_gold in protein:
            is_match = True
        elif protein.lower() in protein_gold.lower() or protein_gold.lower() in protein.lower():
            is_match = True
        else:
            dl_obj = dl.SequenceMatcher(None, protein, protein_gold)
            curr_ratio = dl_obj.quick_ratio()
            if curr_ratio > min_ratio:
                is_match = True
    #
    return is_match


def match_protein_name_with_gold_list(protein, proteins_gold_list):
    is_match = False
    for curr_gold_protein in proteins_gold_list:
        if match_protein_name(protein, curr_gold_protein):
            is_match = True
            break
    return is_match

def filter_protein_nodes_list(protein_nodes_list, proteins_filter_list):
    if protein_nodes_list is None or not protein_nodes_list:
        return protein_nodes_list
    new_protein_nodes_list = []
    for curr_protein_node in protein_nodes_list:
        if curr_protein_node is None:
            continue
        #
        curr_protein_name = curr_protein_node.name
        if match_protein_name_with_gold_list(curr_protein_name, proteins_filter_list):
            new_protein_nodes_list.append(curr_protein_node)
    return new_protein_nodes_list


def get_lower_case_protein_names_list_from_model_map(protein_name_idlist_map):
    list_of_protein_names_in_model = protein_name_idlist_map.keys()
    # print 'list_of_protein_names_in_model', list_of_protein_names_in_model
    for i in range(len(list_of_protein_names_in_model)):
        list_of_protein_names_in_model[i] = list_of_protein_names_in_model[i].lower()
    # print 'after lower case, list_of_protein_names_in_model', list_of_protein_names_in_model
    return list_of_protein_names_in_model


def is_site(state_str):
    is_site_candidate = False
    if 'tyros' in state_str.lower():
        is_site_candidate = True
    elif 'serin' in state_str.lower():
        is_site_candidate = True
    elif 'threon' in state_str.lower():
        is_site_candidate = True
    elif not site_re.sub('', state_str.upper()):
        is_site_candidate = True
    elif not mutant_re.sub('', state_str.upper()):
        is_site_candidate = True
    elif not number_only_re.sub('', state_str):
        is_site_candidate = True
    return is_site_candidate


def is_valid_protein_node(protein_node):
    if cg.is_identifier_required:
        print 'validating protein identifier', protein_node.identifier
        if protein_node.identifier is None:
            return False
    protein_name = protein_node.name
    print 'validating protein name ', protein_name
    if protein_name is None or not protein_name.strip():
        return False
    is_valid = True
    if '[' in protein_name:
        is_valid = False
    elif ']' in protein_name:
        is_valid = False
    elif 'tyros' in protein_name.lower():
        is_valid = False
    elif 'serin' in protein_name.lower():
        is_valid = False
    elif 'threon' in protein_name.lower():
        is_valid = False
    elif not site_re.sub('', protein_name.upper()):
        is_valid = False
    elif not mutant_re.sub('', protein_name.upper()):
        is_valid = False
    elif not number_only_re.sub('', protein_name):
        is_valid = False
    elif cvp.is_wordvec_fr_protein:
        word_vec_protein_name = get_wordvector(protein_name)
        if word_vec_protein_name is None:
            is_valid = False
            print 'could not get word vector for protein name {} from amrs'.format(protein_name)
    print '{} is {}'.format(protein_name, 'valid' if is_valid else 'invalid')
    return is_valid


def is_wrong_proteins_pair_fr_interaction(protein1_node, protein2_node):
    if protein1_node is None and protein2_node is None:
        return True
    if protein1_node is None or protein2_node is None:
        return False
    if protein1_node.name.lower() in protein2_node.name.lower():
        return True
    if protein2_node.name.lower() in protein1_node.name.lower():
        return True
    if cg.is_identifier_required:
        if protein1_node.identifier is None and protein2_node.identifier is None:
            return True
        if protein1_node.identifier is None or protein2_node.identifier is None:
            return False
        if protein1_node.identifier == protein2_node.identifier:
            return True
    return False


def get_bind_concept_path_nodes(curr_catalyst_path_nodes, curr_domain_path_nodes, curr_domain2_path_nodes, catalyst_prob_vec=None, domain_prob_vec=None,
                                domain2_prob_vec=None):
    if curr_domain_path_nodes is None or curr_domain2_path_nodes is None:
        raise AssertionError('domain path nodes or domain2 path nodes should not be None')
    if curr_catalyst_path_nodes is not None and catalyst_prob_vec is not None and domain_prob_vec is not None and domain2_prob_vec is not None:
        catalyst_only_prob = np.array([catalyst_prob_vec[1], domain_prob_vec[1], domain2_prob_vec[1]])
        if catalyst_only_prob.argmax() != 0:
            #switch names for the path nodes
            if catalyst_only_prob.argmax() == 1:
                temp = curr_domain_path_nodes
                curr_domain_path_nodes = curr_catalyst_path_nodes
                curr_catalyst_path_nodes = temp
                temp = None
            elif catalyst_only_prob.argmax() == 2:
                temp = curr_domain2_path_nodes
                curr_domain2_path_nodes = curr_catalyst_path_nodes
                curr_catalyst_path_nodes = temp
                temp = None
            else:
                raise AssertionError('contradiction with the outer if condition')
    # todo: deep copy is essential for ensuring right color for nodes
    # todo: though the merging of copy of paths may itself lead to unexpected results, verify that it is not the case or change code
    curr_concept_path_nodes = []
    curr_bind_tuple = [None, None, None, None]
    if curr_catalyst_path_nodes is not None:
        # catalyst
        copy_curr_catalyst_path_nodes = copy.deepcopy(curr_catalyst_path_nodes)
        copy_curr_catalyst_path_nodes[-1].color = 'green'
        #
        curr_bind_tuple[1] = copy.copy(copy_curr_catalyst_path_nodes[-1])
        curr_bind_tuple[1].reset_parent_children_none()
        #
        curr_concept_path_nodes += copy_curr_catalyst_path_nodes
        copy_curr_catalyst_path_nodes = None
    # domain
    copy_curr_domain_path_nodes = copy.deepcopy(curr_domain_path_nodes)
    copy_curr_domain_path_nodes[-1].color = '#976850'
    #
    curr_bind_tuple[2] = copy.copy(copy_curr_domain_path_nodes[-1])
    curr_bind_tuple[2].reset_parent_children_none()
    #
    if curr_catalyst_path_nodes is not None:
        curr_concept_path_nodes += copy_curr_domain_path_nodes[1:]
    else:
        curr_concept_path_nodes += copy_curr_domain_path_nodes
    copy_curr_domain_path_nodes = None
    # domain 2
    copy_curr_domain2_path_nodes = copy.deepcopy(curr_domain2_path_nodes)
    copy_curr_domain2_path_nodes[-1].color = '#976856'
    #
    curr_bind_tuple[3] = copy.copy(copy_curr_domain2_path_nodes[-1])
    curr_bind_tuple[3].reset_parent_children_none()
    #
    curr_concept_path_nodes += copy_curr_domain2_path_nodes[1:]
    curr_concept_path_nodes[0].color = 'blue'
    #
    curr_bind_tuple[0] = copy.copy(curr_concept_path_nodes[0])
    curr_bind_tuple[0].reset_parent_children_none()
    #
    curr_concept_path_nodes = remove_duplicates_based_on_ids(curr_concept_path_nodes)
    #disconnecting path nodes from other nodes (connected through parents and children)
    curr_concept_path_nodes = prune_non_path_nodes_references_frm_subgraph(curr_concept_path_nodes)
    #
    return curr_concept_path_nodes, tuple(curr_bind_tuple)


def get_state_change_concept_path_nodes(curr_catalyst_path_nodes, curr_domain_path_nodes, catalyst_prob_vec=None, domain_prob_vec=None):
    if curr_domain_path_nodes is None:
        raise AssertionError('domain path nodes can not be None')
    if curr_catalyst_path_nodes is not None and catalyst_prob_vec is not None and domain_prob_vec is not None:
        catalyst_only_prob = np.array([catalyst_prob_vec[1], domain_prob_vec[1]])
        if catalyst_only_prob.argmax() == 1:
            #switch names for the path nodes
            temp = curr_domain_path_nodes
            curr_domain_path_nodes = curr_catalyst_path_nodes
            curr_catalyst_path_nodes = temp
            temp = None
    curr_concept_path_nodes = []
    curr_state_change_tuple = [None, None, None]
    if curr_catalyst_path_nodes is not None:
        # catalyst
        copy_curr_catalyst_path_nodes = copy.deepcopy(curr_catalyst_path_nodes)
        copy_curr_catalyst_path_nodes[-1].color = 'green'
        #
        curr_state_change_tuple[1] = copy.copy(copy_curr_catalyst_path_nodes[-1])
        curr_state_change_tuple[1].reset_parent_children_none()
        #
        curr_concept_path_nodes += copy_curr_catalyst_path_nodes
        copy_curr_catalyst_path_nodes = None
    # domain
    copy_curr_domain_path_nodes = copy.deepcopy(curr_domain_path_nodes)
    copy_curr_domain_path_nodes[-1].color = '#976850'
    #
    curr_state_change_tuple[2] = copy.copy(copy_curr_domain_path_nodes[-1])
    curr_state_change_tuple[2].reset_parent_children_none()
    #
    if curr_catalyst_path_nodes is not None:
        curr_concept_path_nodes += copy_curr_domain_path_nodes[1:]
    else:
        curr_concept_path_nodes += copy_curr_domain_path_nodes
    copy_curr_domain_path_nodes = None
    curr_concept_path_nodes[0].color = 'blue'
    #
    curr_state_change_tuple[0] = copy.copy(curr_concept_path_nodes[0])
    curr_state_change_tuple[0].reset_parent_children_none()
    #
    curr_concept_path_nodes = remove_duplicates_based_on_ids(curr_concept_path_nodes)
    #disconnecting path nodes from other nodes (connected through parents and children)
    curr_concept_path_nodes = prune_non_path_nodes_references_frm_subgraph(curr_concept_path_nodes)
    #
    return curr_concept_path_nodes, tuple(curr_state_change_tuple)


def gen_catalyst_domain_paths_frm_amr_concept_nd_wr_dot_pdf(amr_dot_file_path, is_joint_path=False, protein_name_idlist_map=None,
                                                            kernel_non_joint_trained_model_obj=None, proteins_filter_list=None):
    def add_joint_path_subgraph(curr_concept_node, curr_path_nodes, curr_interaction_tuple, curr_sentence, count, nodes_list_map,
                                interaction_tuples_map, sentences_map, triplet_str_key_tuple_map=None, curr_triplet_str_key_tuple=None):
        #
        print '***********************************************************'
        print 'curr_triplet_str_key_tuple in joint ', curr_triplet_str_key_tuple
        print 'triplet_str_key_tuple_map.keys()', triplet_str_key_tuple_map.keys() if triplet_str_key_tuple_map is not None else None
        print 'len(curr_path_nodes)', len(curr_path_nodes)
        print '***********************************************************'
        #
        if curr_path_nodes[0].name != curr_concept_node.name or curr_path_nodes[0].id != curr_concept_node.id:
            raise AssertionError
        file_path = None
        if triplet_str_key_tuple_map is not None and triplet_str_key_tuple_map:
            if curr_triplet_str_key_tuple in triplet_str_key_tuple_map:
                if len(curr_path_nodes) >= triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['num_nodes']:
                    print 'this interaction triplet str with shortest subgraph instance is already encountered, so not adding this one ..'
                    return
                else:
                    print 'this is shorter than previous one, so replacing previous one by using previous file path ..'
                    file_path = triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['file_path']
                    assert file_path is not None
            else:
                print 'this interaction triplet str is encountered first time.'
        path_sentence = 'concept is ' + curr_concept_node.name
        print path_sentence
        path_sentence += '\n' + 'org:-' + org_sentence
        #setting is_root fr nodes in a path
        for curr_path_node in curr_path_nodes:
            curr_path_node.is_root = False
        curr_path_nodes[0].is_root = True #first node is supposed to be concept node
        if file_path is None:
            file_path = amr_dot_file_path+'_joint_'+str(count)
        else:
            assert triplet_str_key_tuple_map is not None and triplet_str_key_tuple_map
            assert curr_triplet_str_key_tuple in triplet_str_key_tuple_map
            assert len(curr_path_nodes) < triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['num_nodes']
        print file_path
        nodes_to_dot(curr_path_nodes, file_path, path_sentence)
        nodes_list_map[file_path] = curr_path_nodes
        interaction_tuples_map[file_path] = curr_interaction_tuple
        sentences_map[file_path] = curr_sentence
        if triplet_str_key_tuple_map is not None:
            if curr_triplet_str_key_tuple not in triplet_str_key_tuple_map:
                triplet_str_key_tuple_map[curr_triplet_str_key_tuple] = {}
            triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['num_nodes'] = len(curr_path_nodes)
            triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['file_path'] = file_path

    nodes, org_sentence = build_nodes_tree_from_amr_dot_file(amr_dot_file_path)
    nodes = preprocess_nodes_fr_kernel_based_extraction(nodes)
    #
    concept_similarity_tol = 0.02
    protein_nodes = []
    complex_form_concept_nodes = []
    state_change_concept_nodes = []
    for curr_node in nodes.values():
        print 'curr_node', curr_node
        if (not concept_regexp.sub('', curr_node.name)) or (curr_node.name == curr_node.type):
            if not is_invalid_state_or_interaction_type(concept_num_regexp.sub('', curr_node.name), (list_of_invalid_state_or_interaction_type+list_of_invalid_interaction_type)):
                is_similar_to_interaction, interaction_similarity = is_word_similar(concept_num_regexp.sub('', curr_node.name), interaction_labels)
                is_similar_to_complex_form, complex_form_similarity = is_word_similar(concept_num_regexp.sub('', curr_node.name), complex_labels)
                if is_similar_to_interaction and is_similar_to_complex_form:
                    if abs(interaction_similarity-complex_form_similarity) < concept_similarity_tol:
                        state_change_concept_nodes.append(curr_node)
                        print 'added node to state nodes list'
                        complex_form_concept_nodes.append(curr_node)
                        print 'added node to complex formation types list'
                    else:
                        if interaction_similarity > complex_form_similarity:
                            state_change_concept_nodes.append(curr_node)
                            print 'added node to state nodes list'
                        else:
                            complex_form_concept_nodes.append(curr_node)
                            print 'added node to complex formation types list'
                elif is_similar_to_interaction:
                        state_change_concept_nodes.append(curr_node)
                        print 'added node to state nodes list'
                elif is_similar_to_complex_form:
                        complex_form_concept_nodes.append(curr_node)
                        print 'added node to complex formation types list'
        else:
            if is_valid_protein_node(curr_node):
                is_add_protein_node = False
                if is_word_similar(curr_node.type, protein_labels)[0]:
                    is_add_protein_node = True
                elif is_word_similar(curr_node.name, protein_labels)[0]:
                    is_add_protein_node = True
                else:
                    if cvp.is_protein_name_wordvec_similar_proteins_in_model \
                            and is_word_similar(curr_node.name, bmo.bm_obj.protein_identifier_list_map.keys())[0]:
                        is_add_protein_node = True
                if is_add_protein_node:
                    if debug:
                        print 'adding node for protein ', curr_node.name
                    protein_nodes.append(curr_node)
    #
    concept_nodes = my_util.unique_list(state_change_concept_nodes + complex_form_concept_nodes)
    # print 'concept_nodes', concept_nodes
    # print 'state_change_concept_nodes', state_change_concept_nodes
    # print 'complex_form_concept_nodes', complex_form_concept_nodes
    if len(concept_nodes) == 0 or len(protein_nodes) == 0:
        return {}, {}, {}
    # filter protein nodes
    if dc.is_darpa and dc.all_proteins_of_interaction_in_model:
        protein_nodes = filter_protein_nodes_of_model(protein_nodes, protein_name_idlist_map)
    if proteins_filter_list is not None:
        protein_nodes = filter_protein_nodes_list(protein_nodes, proteins_filter_list)
    #
    count = 0
    paths = {}
    interaction_tuples_map = {}
    sentences_map = {}
    if ce.is_single_instance_fr_tuple:
        triplet_str_key_tuple_map = {}
    else:
        triplet_str_key_tuple_map = None
    if not is_joint_path:
        for curr_protein_node in protein_nodes:
            for curr_concept_node in concept_nodes:
                count += 1
                path_sentence = 'path from ' + curr_concept_node.name + ' to ' + curr_protein_node.name
                print path_sentence
                path_sentence += '\n' + 'org: ' + org_sentence
                curr_path_nodes = search_shortest_undirected_bfs_path(curr_concept_node, curr_protein_node)
                if curr_path_nodes is not None:
                    curr_path_nodes = copy.deepcopy(curr_path_nodes) #copying nodes an individual amr
                    #setting is_root fr nodes in a path
                    for curr_path_node in curr_path_nodes:
                        curr_path_node.is_root = False
                    curr_path_nodes[0].is_root = True #first node is supposed to be concept node
                    #disconnecting path nodes from other nodes (connected through parents and children)
                    curr_path_nodes = prune_non_path_nodes_references_frm_subgraph(curr_path_nodes)
                    #
                    file_path = amr_dot_file_path+'_'+str(count)
                    nodes_to_dot(curr_path_nodes, file_path, path_sentence)
                    #
                    assert_name_id_match_fr_nodes(curr_path_nodes[0], curr_concept_node)
                    assert_name_id_match_fr_nodes(curr_path_nodes[-1], curr_protein_node)
                    curr_interaction_tuple = (copy.copy(curr_path_nodes[0]), copy.copy(curr_path_nodes[-1]))
                    curr_interaction_tuple[0].reset_parent_children_none()
                    curr_interaction_tuple[1].reset_parent_children_none()
                    #
                    paths[file_path] = curr_path_nodes
                    interaction_tuples_map[file_path] = curr_interaction_tuple
                    sentences_map[file_path] = org_sentence
    else:
        if ce.is_protein_path_lkl_screening:
            prob_min = 0.075
        temp_path_file_name = './temp/temp_path'+str(r.randint(1, 1000000))
        interactions_list = []
        complex_form_list = []
        if kernel_non_joint_trained_model_obj is None and ce.is_protein_path_lkl_screening:
            kernel_non_joint_trained_model_obj = ie.KernelClassifier(is_joint=False)
        for curr_concept_node in concept_nodes:
            if debug:
                print 'curr_concept_node', curr_concept_node
            for curr_catalyst_node in protein_nodes+[None]:
                if debug:
                    print 'curr_catalyst_node', curr_catalyst_node
                curr_catalyst_path_nodes = None
                if curr_catalyst_node is not None:
                    if curr_catalyst_node.name.lower() == curr_concept_node.name.lower():
                        continue
                    curr_catalyst_path_nodes = search_shortest_undirected_bfs_path(curr_concept_node, curr_catalyst_node)
                    if curr_catalyst_path_nodes is None or len(curr_catalyst_path_nodes) > ce.max_num_nodes_subgraph_path:
                        continue
                    else:
                        if ce.is_protein_path_lkl_screening:
                            curr_catalyst_path_node_map = {'path': temp_path_file_name,
                                                           'nodes': te.get_map_frm_list(prune_non_path_nodes_references_frm_subgraph(
                                                               copy.deepcopy(curr_catalyst_path_nodes)))}
                            catalyst_prob_vec = kernel_non_joint_trained_model_obj.infer_frm_svm_saved_classifier(
                                test_amr_graph=curr_catalyst_path_node_map)
                            print 'catalyst_prob_vec for {} is {}'.format(curr_catalyst_node.name, catalyst_prob_vec)
                            if catalyst_prob_vec[1] < prob_min:
                                continue
                        else:
                            catalyst_prob_vec = None
                        # todo: see probabilty vec and if above threshold for catalyst case, then only consider it as potential candidate
                for curr_domain_node in protein_nodes:
                    if debug:
                        print 'curr_domain_node', curr_domain_node
                    if curr_domain_node is None:
                        if debug:
                            print 'A'
                        continue
                    elif curr_domain_node == curr_catalyst_node or curr_domain_node == curr_concept_node:
                        if debug:
                            print 'A1'
                        continue
                    elif curr_catalyst_node is not None and (curr_domain_node.name.lower() == curr_catalyst_node.name.lower()):
                        if debug:
                            print 'A2'
                        continue
                    elif curr_domain_node.name.lower() == curr_concept_node.name.lower():
                        if debug:
                            print 'A3'
                        continue
                    elif is_wrong_proteins_pair_fr_interaction(curr_catalyst_node, curr_domain_node) and cd.is_darpa:
                        if debug:
                            print 'A0'
                        continue
                    else:
                        if debug:
                            print 'A4'
                        curr_domain_path_nodes = search_shortest_undirected_bfs_path(curr_concept_node, curr_domain_node)
                        if curr_domain_path_nodes is None or len(curr_domain_path_nodes) > ce.max_num_nodes_subgraph_path:
                            if debug:
                                print 'A5'
                            continue
                    if curr_concept_node in complex_form_concept_nodes:
                        if ce.is_protein_path_lkl_screening:
                            curr_domain_path_nodes_map = {'path': temp_path_file_name,
                                                          'nodes': te.get_map_frm_list(prune_non_path_nodes_references_frm_subgraph(
                                                              copy.deepcopy(curr_domain_path_nodes)))}
                            domain_prob_vec = kernel_non_joint_trained_model_obj.infer_frm_svm_saved_classifier(
                                test_amr_graph=curr_domain_path_nodes_map)
                            print 'domain_prob_vec is ', domain_prob_vec
                            if domain_prob_vec[2] < prob_min:
                                continue
                        else:
                            domain_prob_vec = None
                        for curr_domain_node2 in protein_nodes:
                            if debug:
                                print 'curr_domain_node2', curr_domain_node2
                            if curr_domain_node2 is None:
                                continue
                            elif curr_domain_node2 in [curr_catalyst_node, curr_domain_node, curr_concept_node]:
                                continue
                            elif curr_domain_node2.name.lower() in [curr_domain_node.name.lower(), curr_concept_node.name.lower()]:
                                continue
                            elif curr_catalyst_node is not None and (curr_domain_node2.name.lower() == curr_catalyst_node.name.lower()):
                                continue
                            elif is_wrong_proteins_pair_fr_interaction(curr_catalyst_node, curr_domain_node2):
                                continue
                            elif is_wrong_proteins_pair_fr_interaction(curr_domain_node, curr_domain_node2):
                                continue
                            else:
                                if dc.is_darpa and dc.at_least_one_protein_of_interaction_in_model:
                                    curr_proteins_in_complex_list = [curr_catalyst_node, curr_domain_node, curr_domain_node2]
                                    curr_model_filtered_proteins_in_complex_list = filter_protein_nodes_of_model(curr_proteins_in_complex_list,
                                                                                                                 protein_name_idlist_map)
                                    if curr_model_filtered_proteins_in_complex_list is None or not curr_model_filtered_proteins_in_complex_list:
                                        continue
                                curr_domain2_path_nodes = search_shortest_undirected_bfs_path(curr_concept_node, curr_domain_node2)
                                if curr_domain2_path_nodes is None or len(curr_domain2_path_nodes) > ce.max_num_nodes_subgraph_path:
                                    continue
                                else:
                                    # do not select duplicates
                                    curr_entities_set = tuple([curr_concept_node, curr_catalyst_node, curr_domain_node, curr_domain_node2])
                                    if curr_entities_set in complex_form_list:
                                        continue
                                    if ce.is_protein_path_lkl_screening:
                                        curr_domain2_path_nodes_map = {'path': temp_path_file_name,
                                                                       'nodes': te.get_map_frm_list(prune_non_path_nodes_references_frm_subgraph(
                                                                           copy.deepcopy(curr_domain2_path_nodes)))}
                                        domain2_prob_vec = kernel_non_joint_trained_model_obj.infer_frm_svm_saved_classifier(
                                            test_amr_graph=curr_domain2_path_nodes_map)
                                        print 'domain2_prob_vec is ', domain2_prob_vec
                                        if domain2_prob_vec[2] < prob_min:
                                            continue
                                    else:
                                        domain2_prob_vec = None
                                    # get concept path nodes here by calling the inner function
                                    curr_concept_path_nodes, curr_bind_tuple = get_bind_concept_path_nodes(
                                        curr_catalyst_path_nodes, curr_domain_path_nodes, curr_domain2_path_nodes, catalyst_prob_vec,
                                        domain_prob_vec, domain2_prob_vec)
                                    print 'binding interaction'
                                    print (curr_bind_tuple[0].name, curr_bind_tuple[1].name if curr_bind_tuple[1] is not None else None,
                                           curr_bind_tuple[2].name, curr_bind_tuple[3].name)
                                    #
                                    if len(curr_concept_path_nodes) > ce.max_num_nodes_joint_subgraph_fr_complex_interaction:
                                        print 'not considering the subgraph since the graph is too large'
                                        continue
                                    #
                                    assert_name_id_match_fr_nodes(curr_concept_path_nodes[0], curr_concept_node)
                                    #
                                    if ce.is_single_instance_fr_tuple:
                                        curr_triplet_str_key_tuple = (curr_concept_node.name,
                                                                      curr_catalyst_node.name if curr_catalyst_node is not None else None,
                                                                      curr_domain_node.name, curr_domain_node2.name)
                                    else:
                                        curr_triplet_str_key_tuple = None
                                        assert triplet_str_key_tuple_map is None
                                    add_joint_path_subgraph(curr_concept_node, curr_concept_path_nodes, curr_bind_tuple, org_sentence, count, paths,
                                                            interaction_tuples_map, sentences_map,
                                                            triplet_str_key_tuple_map=triplet_str_key_tuple_map,
                                                            curr_triplet_str_key_tuple=curr_triplet_str_key_tuple)
                                    #
                                    complex_form_list.append(curr_entities_set)
                                    count += 1
                    if curr_concept_node in state_change_concept_nodes:
                        # print 'hello in state change block debugging ...'
                        if debug:
                            print 'A6'
                        if dc.is_darpa and dc.at_least_one_protein_of_interaction_in_model:
                            curr_proteins_in_state_change_list = [curr_catalyst_node, curr_domain_node]
                            curr_model_filtered_proteins_in_state_change_in_list = \
                                filter_protein_nodes_of_model(curr_proteins_in_state_change_list, protein_name_idlist_map)
                            if curr_model_filtered_proteins_in_state_change_in_list is None or \
                                    not curr_model_filtered_proteins_in_state_change_in_list:
                                if debug:
                                    print 'A7'
                                continue
                        curr_entities_set = tuple([curr_concept_node, curr_catalyst_node, curr_domain_node])
                        if curr_entities_set in interactions_list:
                            print 'curr_entities_set already encountered.'
                            if debug:
                                print 'interactions_list', interactions_list
                            continue
                        if ce.is_protein_path_lkl_screening:
                            curr_domain_path_nodes_map = {'path': temp_path_file_name,
                                                          'nodes': te.get_map_frm_list(prune_non_path_nodes_references_frm_subgraph(
                                                              copy.deepcopy(curr_domain_path_nodes)))}
                            domain_prob_vec = kernel_non_joint_trained_model_obj.infer_frm_svm_saved_classifier(
                                test_amr_graph=curr_domain_path_nodes_map)
                            if debug:
                                print 'curr_domain_node.name', curr_domain_node.name
                            print 'domain_prob_vec is ', domain_prob_vec
                            if domain_prob_vec[2] < prob_min:
                                continue
                        else:
                            domain_prob_vec = None
                        curr_concept_path_nodes, curr_state_change_tuple = get_state_change_concept_path_nodes(
                            curr_catalyst_path_nodes, curr_domain_path_nodes, catalyst_prob_vec, domain_prob_vec)
                        #
                        if len(curr_concept_path_nodes) > ce.max_num_nodes_joint_subgraph_fr_state_change:
                            print 'not considering the subgraph since the graph is too large'
                            continue
                        #
                        assert_name_id_match_fr_nodes(curr_concept_path_nodes[0], curr_concept_node)
                        #
                        if ce.is_single_instance_fr_tuple:
                            curr_triplet_str_key_tuple = (curr_concept_node.name,
                                                          curr_catalyst_node.name if curr_catalyst_node is not None else None,
                                                          curr_domain_node.name)
                        else:
                            curr_triplet_str_key_tuple = None
                            assert triplet_str_key_tuple_map is None
                        add_joint_path_subgraph(curr_concept_node, curr_concept_path_nodes, curr_state_change_tuple, org_sentence, count,
                                                paths, interaction_tuples_map, sentences_map,
                                                triplet_str_key_tuple_map=triplet_str_key_tuple_map,
                                                curr_triplet_str_key_tuple=curr_triplet_str_key_tuple)
                        #
                        interactions_list.append(curr_entities_set)
                        count += 1
    print 'complex_form_list', complex_form_list
    print 'interactions_list', interactions_list
    return paths, interaction_tuples_map, sentences_map


def is_invalid_state_or_interaction_type(curr_str, list_of_invalid_str):
    is_invalid = False
    for curr_invalid_str in list_of_invalid_str:
        if curr_invalid_str in curr_str or curr_str in curr_invalid_str:
            is_invalid = True
            break
    return is_invalid


def preprocess_nodes_fr_kernel_based_extraction(nodes):
    # sequence is crucial
    nodes = simplify_nodes_tree_names(nodes)
    if cd.is_darpa:
        map_entity_types_to_darpa_types(nodes)
    nodes = simplify_nodes_tree_identifiers(nodes)
    nodes = simplify_nodes_tree_mod(nodes)
    return nodes


def map_entity_types_to_darpa_types(nodes):
    if isinstance(nodes, dict):
        nodes_list = nodes.values()
    elif isinstance(nodes, list):
        nodes_list = nodes
    else:
        raise AssertionError
    for curr_node in nodes_list:
        if concept_regexp.sub('', curr_node.name) and curr_node.name != curr_node.type:
            darpa_entity_type = dptm.get_darpa_protein_type(curr_node.type)
            if darpa_entity_type is not None:
                curr_node.type = darpa_entity_type


def gen_concept_state_paths_frm_amr_protein_nd_wr_dot_pdf(amr_dot_file_path, org_concept_node=None, org_protein_node=None,
                                                          kernel_non_joint_trained_model_obj=None):
    def get_joint_path_nodes(curr_concept_to_protein_path_nodes, curr_protein_to_state_path_nodes):
        curr_joint_path_nodes = [] #first node of joint path should be the protein node not concept or state node
        curr_protein_concept_state_tuple = [None, None, None]
        #
        copy_curr_protein_to_concept_path_nodes = copy.deepcopy(curr_concept_to_protein_path_nodes)
        copy_curr_protein_to_concept_path_nodes.reverse()
        copy_curr_protein_to_concept_path_nodes[-1].color = 'red'
        #
        org_concept_node_ref = copy_curr_protein_to_concept_path_nodes[-1]
        org_protein_node_ref = copy_curr_protein_to_concept_path_nodes[0]
        #
        curr_protein_concept_state_tuple[1] = copy.copy(copy_curr_protein_to_concept_path_nodes[-1])
        curr_protein_concept_state_tuple[1].reset_parent_children_none()
        curr_protein_concept_state_tuple[1].color = '#939393'
        #
        curr_joint_path_nodes += copy_curr_protein_to_concept_path_nodes
        copy_curr_protein_to_concept_path_nodes = None
        #
        copy_curr_protein_to_state_path_nodes = copy.deepcopy(curr_protein_to_state_path_nodes)
        copy_curr_protein_to_state_path_nodes[-1].color = '#976850'
        #
        org_state_node_ref = copy_curr_protein_to_state_path_nodes[-1]
        #
        curr_protein_concept_state_tuple[2] = copy.copy(copy_curr_protein_to_state_path_nodes[-1])
        curr_protein_concept_state_tuple[2].reset_parent_children_none()
        curr_protein_concept_state_tuple[2].color = '#939393'
        #
        curr_joint_path_nodes += copy_curr_protein_to_state_path_nodes[1:]
        copy_curr_protein_to_state_path_nodes = None
        curr_joint_path_nodes[0].color = 'green'
        #
        curr_protein_concept_state_tuple[0] = copy.copy(curr_joint_path_nodes[0])
        curr_protein_concept_state_tuple[0].reset_parent_children_none()
        curr_protein_concept_state_tuple[0].color = '#939393'
        #
        if org_protein_node_ref.is_parent(org_concept_node_ref):
            org_concept_node_ref.add_parent_child_relationship(org_protein_node_ref, pssse.relatedToConcept_of)
        else:
            org_protein_node_ref.add_parent_child_relationship(org_concept_node_ref, pssse.relatedToConcept)
        if org_protein_node_ref.is_parent(org_state_node_ref):
            org_state_node_ref.add_parent_child_relationship(org_protein_node_ref, pssse.hasState_of)
        else:
            org_protein_node_ref.add_parent_child_relationship(org_state_node_ref, pssse.hasState)
        #
        curr_joint_path_nodes = remove_duplicates_based_on_ids(curr_joint_path_nodes)
        #disconnecting path nodes from other nodes (connected through parents and children)
        curr_joint_path_nodes = prune_non_path_nodes_references_frm_subgraph(curr_joint_path_nodes)
        #
        curr_joint_path_nodes = centralize_amr_at_root_node(curr_joint_path_nodes)
        #
        return curr_joint_path_nodes, tuple(curr_protein_concept_state_tuple)

    def add_joint_path_subgraph(curr_protein_node, curr_path_nodes, curr_tuple, curr_sentence, count, nodes_list_map, tuples_map,
                                sentences_map, amr_subgraph_file_path_suffix, triplet_str_key_tuple_map=None,
                                curr_triplet_str_key_tuple=None):
        #
        print '***********************************************************'
        print 'curr_triplet_str_key_tuple for state', curr_triplet_str_key_tuple
        print 'triplet_str_key_tuple_map.keys() for state', triplet_str_key_tuple_map.keys() if triplet_str_key_tuple_map is not None else None
        print 'len(curr_path_nodes)', len(curr_path_nodes)
        print '***********************************************************'
        #
        if curr_path_nodes[0].name != curr_protein_node.name or curr_path_nodes[0].id != curr_protein_node.id:
            raise AssertionError
        #
        file_path = None
        #
        if triplet_str_key_tuple_map is not None and triplet_str_key_tuple_map:
            if curr_triplet_str_key_tuple in triplet_str_key_tuple_map:
                if len(curr_path_nodes) >= triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['num_nodes']:
                    print 'a shorter path for this state from protein already exist, so not adding this one'
                    return
                else:
                    print 'this is shorter than previous one, so replacing previous one by using previous file path ..'
                    file_path = triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['file_path']
                    assert file_path is not None
        #
        path_sentence = 'protein is ' + curr_protein_node.name
        print path_sentence
        path_sentence += '\n' + 'org:-' + org_sentence
        #setting is_root fr nodes in a path
        for curr_path_node in curr_path_nodes:
            curr_path_node.is_root = False
        curr_path_nodes[0].is_root = True #first node is supposed to be concept node
        #
        if file_path is None:
            file_path = amr_dot_file_path+'_state_'+amr_subgraph_file_path_suffix+'_'+str(count)
        else:
            assert triplet_str_key_tuple_map is not None and triplet_str_key_tuple_map
            assert curr_triplet_str_key_tuple in triplet_str_key_tuple_map
            assert len(curr_path_nodes) < triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['num_nodes']
        print file_path
        #
        nodes_to_dot(curr_path_nodes, file_path, path_sentence)
        nodes_to_dot(centralize_amr_at_root_node(curr_path_nodes), file_path+'_cc', path_sentence)
        #
        nodes_list_map[file_path] = curr_path_nodes
        tuples_map[file_path] = curr_tuple
        sentences_map[file_path] = curr_sentence
        #
        if triplet_str_key_tuple_map is not None:
            if curr_triplet_str_key_tuple not in triplet_str_key_tuple_map:
                triplet_str_key_tuple_map[curr_triplet_str_key_tuple] = {}
            triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['num_nodes'] = len(curr_path_nodes)
            triplet_str_key_tuple_map[curr_triplet_str_key_tuple]['file_path'] = file_path

    #
    assert ((org_concept_node is None and org_protein_node is None) or (org_concept_node is not None and org_protein_node is not None))
    if org_concept_node is not None and org_protein_node is not None:
        is_specific = True
    #
    # sequence is crucial
    nodes, org_sentence = build_nodes_tree_from_amr_dot_file(amr_dot_file_path)
    nodes = preprocess_nodes_fr_kernel_based_extraction(nodes)
    #
    concept_similarity_tol = 0.02
    protein_nodes = []
    complex_form_concept_nodes = []
    state_change_concept_nodes = []
    state_nodes = []
    for curr_node in nodes.values():
        if (not concept_regexp.sub('', curr_node.name)) or (curr_node.name == curr_node.type):
            is_similar_to_interaction, interaction_similarity = is_word_similar(concept_num_regexp.sub('', curr_node.name), interaction_labels)
            if is_similar_to_interaction:
                if not is_invalid_state_or_interaction_type(concept_num_regexp.sub('', curr_node.name), list_of_invalid_state_or_interaction_type):
                    state_nodes.append(curr_node)
            else:
                is_similar_to_protein_state, _ = is_word_similar(concept_num_regexp.sub('', curr_node.name), protein_part_labels)
                if not is_similar_to_protein_state:
                    is_similar_to_protein_state, _ = is_word_similar(concept_num_regexp.sub('', curr_node.name), state_labels)
                if is_similar_to_protein_state:
                    if not is_invalid_state_or_interaction_type(concept_num_regexp.sub('', curr_node.name), list_of_invalid_state_or_interaction_type):
                        state_nodes.append(curr_node)
            is_similar_to_complex_form, complex_form_similarity = is_word_similar(concept_num_regexp.sub('', curr_node.name), complex_labels)
            if not is_invalid_state_or_interaction_type(concept_num_regexp.sub('', curr_node.name), (list_of_invalid_state_or_interaction_type+list_of_invalid_interaction_type)):
                if is_similar_to_interaction and is_similar_to_complex_form:
                    if abs(interaction_similarity-complex_form_similarity) < concept_similarity_tol:
                        state_change_concept_nodes.append(curr_node)
                        complex_form_concept_nodes.append(curr_node)
                    else:
                        if interaction_similarity > complex_form_similarity:
                            state_change_concept_nodes.append(curr_node)
                        else:
                            complex_form_concept_nodes.append(curr_node)
                elif is_similar_to_interaction:
                        state_change_concept_nodes.append(curr_node)
                elif is_similar_to_complex_form:
                        complex_form_concept_nodes.append(curr_node)
        else:
            is_add_protein_node = None
            if is_valid_protein_node(curr_node):
                is_add_protein_node = False
                if is_word_similar(curr_node.type, protein_labels)[0]:
                    is_add_protein_node = True
                elif is_word_similar(curr_node.name, protein_labels)[0]:
                    is_add_protein_node = True
                else:
                    if cvp.is_protein_name_wordvec_similar_proteins_in_model \
                            and is_word_similar(curr_node.name, bmo.bm_obj.protein_identifier_list_map.keys())[0]:
                        is_add_protein_node = True
                if is_add_protein_node:
                    if debug:
                        print 'adding node for protein ', curr_node.name
                    protein_nodes.append(curr_node)
            if is_add_protein_node is None or not is_add_protein_node:
                if is_word_similar(curr_node.type, protein_part_labels)[0]:
                    state_nodes.append(curr_node)
                elif is_word_similar(curr_node.name, protein_part_labels)[0]:
                    state_nodes.append(curr_node)
                elif is_site(curr_node.name):
                    state_nodes.append(curr_node)
    concept_nodes = my_util.unique_list(state_change_concept_nodes + complex_form_concept_nodes)
    # print 'concept_nodes', concept_nodes
    # print 'state_change_concept_nodes', state_change_concept_nodes
    # print 'complex_form_concept_nodes', complex_form_concept_nodes
    state_change_concept_nodes = None
    complex_form_concept_nodes = None
    # filter list of concept nodes and protein nodes as per given protein_str and concept_str
    print 'is_specific', is_specific
    if is_specific:
        print 'org_protein_node', org_protein_node
        print 'org_concept_node', org_concept_node
        amr_subgraph_file_path_suffix = '_'+org_concept_node.id + '_' + org_protein_node.id
        # filtering concept nodes
        new_concept_nodes = []
        for curr_concept_node in concept_nodes:
            if curr_concept_node.id == org_concept_node.id:
                new_concept_nodes.append(curr_concept_node)
        concept_nodes = new_concept_nodes
        assert concept_nodes, 'No node found with matching concept_str {}'.format(org_concept_node)
        # filtering protein nodes
        new_protein_nodes = []
        for curr_protein_node in protein_nodes:
            if curr_protein_node.id == org_protein_node.id:
                new_protein_nodes.append(curr_protein_node)
        protein_nodes = new_protein_nodes
        assert protein_nodes, 'No node found with matching protein_str {}'.format(org_protein_node)
        #
        new_state_nodes = []
        for curr_state_node in state_nodes:
            print 'curr_state_node', curr_state_node
            if curr_state_node.name == org_concept_node.name or curr_state_node.name == org_protein_node.name:
                print 'skipping this one ...'
                continue
            new_state_nodes.append(curr_state_node)
        state_nodes = new_state_nodes
        #
    else:
        amr_subgraph_file_path_suffix = ''
    if (not concept_nodes) or (not protein_nodes) or (not state_nodes):
        return {}, {}, {}
    count = 0
    paths = {}
    tuples_map = {}
    sentences_map = {}
    if ce.is_single_instance_fr_tuple and is_specific:
        triplet_str_key_tuple_map = {}
    else:
        triplet_str_key_tuple_map = None
    #
    prob_min = 0.15
    temp_path_file_name = './temp/temp_state_path'+str(r.randint(1, 1000000))
    protein_state_list = []
    if kernel_non_joint_trained_model_obj is None:
        kernel_non_joint_trained_model_obj = ie.KernelClassifier(is_joint=False)
    for curr_concept_node in concept_nodes:
        assert (curr_concept_node is not None)
        if debug:
            print 'curr_concept_node', curr_concept_node
        for curr_protein_node in protein_nodes:
            assert (curr_protein_node is not None)
            if debug:
                print 'curr_protein_node', curr_protein_node
            if curr_protein_node == curr_concept_node or curr_protein_node.name == curr_concept_node.name:
                continue
            curr_concept_to_protein_path_nodes = search_shortest_undirected_bfs_path(curr_concept_node, curr_protein_node)
            if curr_concept_to_protein_path_nodes is None or len(curr_concept_to_protein_path_nodes) > ce.max_num_nodes_subgraph_path:
                continue
            if not is_specific:
                if ce.is_protein_path_lkl_screening:
                    curr_concept_to_protein_path_nodes_map =\
                        {'path': temp_path_file_name,
                         'nodes': te.get_map_frm_list(
                             eliminate_first_order_cycles(
                                 prune_non_path_nodes_references_frm_subgraph(
                                     copy.deepcopy(curr_concept_to_protein_path_nodes))))}
                    protein_prob_vec = kernel_non_joint_trained_model_obj.infer_frm_svm_saved_classifier(test_amr_graph=curr_concept_to_protein_path_nodes_map)
                    if debug:
                        print 'protein_prob_vec is ', protein_prob_vec
                    if (1-protein_prob_vec[0]) < prob_min:
                        continue
            # todo: see probabilty vec and if above threshold for catalyst case, then only consider it as potential candidate
            for curr_state_node in state_nodes:
                assert (curr_state_node is not None)
                if debug:
                    print 'curr_state_node', curr_state_node
                if len(set([curr_concept_node, curr_protein_node, curr_state_node])) < 3:
                    continue
                elif len(set([curr_protein_node.name, curr_state_node.name])) < 2:
                    continue
                else:
                    curr_protein_to_state_path_nodes = search_shortest_undirected_bfs_path(curr_protein_node, curr_state_node)
                    if curr_protein_to_state_path_nodes is None or len(curr_protein_to_state_path_nodes) > ce.max_num_nodes_subgraph_path:
                        continue
                curr_entities_set = tuple([curr_protein_node, curr_concept_node, curr_state_node])
                if curr_entities_set in protein_state_list:
                    continue
                curr_joint_path_nodes, curr_protein_concept_state_tuple =\
                    get_joint_path_nodes(
                        curr_concept_to_protein_path_nodes, curr_protein_to_state_path_nodes)
                #
                assert_name_id_match_fr_nodes(curr_joint_path_nodes[0], curr_protein_node)
                #
                if ce.is_single_instance_fr_tuple:
                    curr_triplet_str_key_tuple = (curr_protein_node.name, curr_concept_node.name, curr_state_node.name)
                else:
                    assert triplet_str_key_tuple_map is None
                    curr_triplet_str_key_tuple = None
                add_joint_path_subgraph(
                    curr_protein_node, curr_joint_path_nodes, curr_protein_concept_state_tuple, org_sentence, count,
                    paths, tuples_map, sentences_map, amr_subgraph_file_path_suffix,
                    curr_triplet_str_key_tuple=curr_triplet_str_key_tuple,
                    triplet_str_key_tuple_map=triplet_str_key_tuple_map)
                #
                protein_state_list.append(curr_entities_set)
                count += 1
    return paths, tuples_map, sentences_map


def is_word_similar(word, word_list, cosine_cut_off=0.85):
    if debug:
        print 'word_list', word_list
    if word in word_list:
        return True, 1
    word_vec = get_wordvector(word)
    if word_vec is None:
        return False, 0
    for curr_word in word_list:
        curr_word_vec = get_wordvector(curr_word)
        if curr_word_vec is None:
            continue
        curr_similarity = gk.get_cosine_similarity(word, curr_word, word_vec, curr_word_vec)
        if curr_similarity > cosine_cut_off:
            return True, curr_similarity
    return False, 0


def prune_non_path_nodes_references_frm_subgraph(path_nodes_list):
    #now, remove the children and parents references which are not part of the final list of nodes
    if debug:
        print '**********************************************************************'
        print 'nodes along path are: '
        for node in path_nodes_list:
            print str(node)
    path_nodes_ids_map = {}
    for node in path_nodes_list:
        path_nodes_ids_map[node.id] = node
    if len(path_nodes_ids_map) != len(path_nodes_list):
        raise AssertionError
    if debug:
        print 'ids of nodes along path are: ', path_nodes_ids_map
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    for node in path_nodes_list:
        if debug:
            print 'node before change: ', node
        children_new = {}
        for children_key in node.children:
            children = node.children[children_key]
            if isinstance(children, list):
                children_list_new_map = {} #using a map instead of list so as to ensure no duplicates (different objects but same key)
                for child in children:
                    if child.id in path_nodes_ids_map:
                        if child.id not in children_list_new_map:
                            children_list_new_map[child.id] = path_nodes_ids_map[child.id]
                            # consistency on parent-child references
                            if not path_nodes_ids_map[child.id].is_parent(node)[0]:
                                path_nodes_ids_map[child.id].add_parent(children_key, node)
                if children_list_new_map:
                    children_new[children_key] = children_list_new_map.values()
            else:
                child = children
                if child.id in path_nodes_ids_map:
                    children_new[children_key] = path_nodes_ids_map[child.id]
                    # consistency on parent-child references
                    if not path_nodes_ids_map[child.id].is_parent(node)[0]:
                        path_nodes_ids_map[child.id].add_parent(children_key, node)
        parents_new = {}
        for parents_key in node.parents:
            parents = node.parents[parents_key]
            parents_list_new_map = {}
            for parent in parents:
                # print 'parent.id ', parent.id
                # print 'parent.name', parent.name
                if parent.id in path_nodes_ids_map:
                    if parent.id not in parents_list_new_map:
                        # print 'adding ...'
                        parents_list_new_map[parent.id] = path_nodes_ids_map[parent.id]
                        # consistency on parent-child references
                        if not path_nodes_ids_map[parent.id].is_child(node)[0]:
                            path_nodes_ids_map[parent.id].add_child(parents_key, node)
            if parents_list_new_map:
                parents_new[parents_key] = parents_list_new_map.values()
        node.reset_hidden_none()
        node.parents = parents_new
        node.children = children_new
        node.is_root = False
        if debug:
            print 'node after change: ', node
    path_nodes_list[0].is_root = True
    if debug:
        print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    return path_nodes_list


def centralize_amr_at_root_node(amr_nodes_list, root_node_id=None):
    if debug:
        print 'here is the list of nodes under concept centralization ...'
    for curr_node in amr_nodes_list:
        if debug:
            print 'curr_node', curr_node
    curr_node = None
    #
    if debug:
        print 'the core algorithm for centralization starts here'
    #
    # concept node at which to centralize is first node in amr_nodes_list
    list_of_prv_nodes_in_queue = []
    list_of_nodes_pair_as_set_from_added_edges = []
    if root_node_id is None:
        list_nodes_in_queue = []
        list_nodes_in_queue.append(amr_nodes_list[0]) #adding concept node to the queue
    else:
        if debug:
            print 'centralizing at a particular root node with id {}'.format(root_node_id)
        list_nodes_in_queue = []
        for curr_node in amr_nodes_list:
            if curr_node.id == root_node_id:
                if debug:
                    print 'node with root id ', curr_node
                list_nodes_in_queue.append(curr_node)
                break
        if not list_nodes_in_queue:
            raise AssertionError
    centralize_amr_at_root_node_recursive(list_of_prv_nodes_in_queue=list_of_prv_nodes_in_queue, list_of_nodes_pair_as_set_from_added_edges=list_of_nodes_pair_as_set_from_added_edges, list_nodes_in_queue=list_nodes_in_queue)
    return amr_nodes_list


def centralize_amr_at_root_node_recursive(list_of_prv_nodes_in_queue, list_of_nodes_pair_as_set_from_added_edges, list_nodes_in_queue):
    # get the node from the front of the queue
    curr_node = list_nodes_in_queue[0]
    list_nodes_in_queue.remove(curr_node)
    list_of_prv_nodes_in_queue.append(curr_node) #this may not be required
    if debug:
        print 'curr_node.name', curr_node.name
    # children of a node remain as it is
    for curr_children_key in curr_node.children:
        if debug:
            print 'curr_children_key', curr_children_key
        curr_children_list = curr_node.children[curr_children_key]
        if isinstance(curr_children_list, list):
            for curr_child in curr_children_list:
                if debug:
                    print 'curr_child', curr_child
                curr_edge_set = set([curr_node, curr_child])
                if curr_edge_set not in list_of_nodes_pair_as_set_from_added_edges:
                    list_of_nodes_pair_as_set_from_added_edges.append(curr_edge_set)
                if curr_child not in list_nodes_in_queue and curr_child not in list_of_prv_nodes_in_queue:
                    list_nodes_in_queue.append(curr_child)
        else:
            curr_child = curr_children_list
            if debug:
                print 'curr_child', curr_child
            curr_edge_set = set([curr_node, curr_child])
            if curr_edge_set not in list_of_nodes_pair_as_set_from_added_edges:
                list_of_nodes_pair_as_set_from_added_edges.append(curr_edge_set)
            if curr_child not in list_nodes_in_queue and curr_child not in list_of_prv_nodes_in_queue:
                list_nodes_in_queue.append(curr_child)
    # parent become children if no edge added between the nodes so far
    curr_node_parent_keys = copy.copy(curr_node.parents.keys())
    for curr_parents_key in curr_node_parent_keys:
        if debug:
            print 'curr_parents_key', curr_parents_key
        if curr_parents_key in curr_node.parents:
            curr_parents_list_copy = copy.copy(curr_node.parents[curr_parents_key])
            for curr_parent in curr_parents_list_copy:
                if curr_parent not in list_nodes_in_queue and curr_parent not in list_of_prv_nodes_in_queue:
                    list_nodes_in_queue.append(curr_parent)
                if curr_parent in curr_node.parents[curr_parents_key]:
                    if debug:
                        print 'curr_parent', curr_parent
                    curr_edge_set = set([curr_node, curr_parent])
                    # apply inverse operation on this parent
                    if curr_edge_set not in list_of_nodes_pair_as_set_from_added_edges:
                        if debug:
                            print 'curr_edge_set', (curr_node.name, curr_parent.name)
                        # inverse key
                        if curr_parents_key.endswith('-of'):
                            new_edge_key = curr_parents_key[:-3]
                        else:
                            new_edge_key = curr_parents_key+'-of'
                        if debug:
                            print 'new_edge_key', new_edge_key
                        # current_parent is no longer parent of current node
                        curr_parent.remove_parent_child_relationship(curr_node, curr_parents_key)
                        # current_parent is now child of current node with new inverse key
                        curr_node.add_parent_child_relationship(curr_parent, new_edge_key)
                        #
                        if debug:
                            print 'after the relinking ...'
                            print 'curr_node is ', curr_node
                            print 'curr_parent is ', curr_parent
                        #
                        list_of_nodes_pair_as_set_from_added_edges.append(curr_edge_set)
                    else:
                        pass
                        #todo: should we remove any edge with parent since it is already there ? I don't think so as of now
                else:
                    raise AssertionError
        else:
            raise AssertionError
    if list_nodes_in_queue:
        centralize_amr_at_root_node_recursive(list_of_prv_nodes_in_queue=list_of_prv_nodes_in_queue, list_of_nodes_pair_as_set_from_added_edges=list_of_nodes_pair_as_set_from_added_edges, list_nodes_in_queue=list_nodes_in_queue)


def remove_duplicates_based_on_ids(nodes_list):
    concept_node_id = nodes_list[0].id
    new_node_map = {}
    for node in nodes_list:
        if node.id != concept_node_id:
            if node.id not in new_node_map:
                new_node_map[node.id] = node
            else:
                if hasattr(node, 'color') and node.color is not None:
                    if (not hasattr(new_node_map[node.id], 'color')) or (new_node_map[node.id].color is None):
                        new_node_map[node.id] = node
    new_nodes_list = new_node_map.values()
    new_nodes_list.insert(0, nodes_list[0])
    return new_nodes_list


def eliminate_first_order_cycles(nodes_list):
    for curr_node in nodes_list:
        children_keys_list = copy.copy(curr_node.children.keys())
        for curr_children_key in children_keys_list:
            children_list = curr_node.children[curr_children_key]
            if not isinstance(children_list, list):
                children_list = [children_list]
            for child in children_list:
                # this is a case of first order cycle
                # to remove the first order cycle, apply inverse operation
                if curr_node in child.create_parent_list():
                    new_child_edge_key = el.get_inverse_of_edge_label(curr_children_key)
                    curr_node.remove_parent_child_relationship(child, curr_children_key)
                    child.add_parent_child_relationship(curr_node, new_child_edge_key)
    return nodes_list


