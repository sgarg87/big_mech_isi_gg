import pydot as pd
import constants_absolute_path as cap
import extract_from_amr_dot as ead
import config_hpcc as ch
from config_console_output import *


def get_name_identifier_map(nodes_list):
    name_identifier_map = {}
    for curr_node in nodes_list:
        if curr_node.identifier is not None:
            name_identifier_map[curr_node.name] = curr_node.identifier
    return name_identifier_map


def run(amr_dot_file, start_amr, end_amr):
    def print_nodes(nodes):
        print 'printing nodes ..........................................................'
        for curr_node in nodes:
            print curr_node

    assert amr_dot_file is not None and amr_dot_file
    assert start_amr is not None
    assert end_amr is not None
    name_identifier_map = {}
    for i in range(start_amr, end_amr+1):
        try:
            curr_amr_dot_file = amr_dot_file + '.' + str(i) + '.dot'
            print 'curr_amr_dot_file:', curr_amr_dot_file
            #
            curr_amr_dot_obj = pd.graph_from_dot_file(cap.absolute_path+curr_amr_dot_file)
            if not ch.is_hpcc:
                curr_amr_dot_obj.write_pdf(cap.absolute_path+curr_amr_dot_file+'.pdf')
            nodes, org_sentence = ead.build_nodes_tree_from_amr_dot_file(curr_amr_dot_file)
            nodes = ead.simplify_nodes_tree_names(nodes)
            nodes = ead.simplify_nodes_tree_identifiers(nodes)
            if not ch.is_hpcc:
                ead.nodes_to_dot(nodes, curr_amr_dot_file+'_si', org_sentence)
            name_identifier_map.update(get_name_identifier_map(nodes.values()))
        except Exception as e:
            # print e.message
            raise
    print 'name_identifier_map', name_identifier_map
    return name_identifier_map


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]))
    else:
        raise AssertionError

