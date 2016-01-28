import copy
import pydot as pd
import constants_absolute_path as cap
from constants import *
from config import *
import util as my_util


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
            return None, list_node_ids_in_subgraph_to_remove

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


