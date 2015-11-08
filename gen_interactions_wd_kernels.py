import config_kernel as ck
import train_extractor as te
import csv
import extract_from_amr_dot as ead
import gen_extractor_features_data as gefd
from config import *
import interactions as it
import infer_extractor as ie
import pydot as pd
import copy
import re
import numpy.random as nprnd
import numpy as np
import constants_absolute_path as cap
import config_hpcc as ch
import math
import kernel_extraction_cutoff_tuned as kect
from config_console_output import *
import file_paths_extraction as fpe
import config_extraction as ce


is_random_state_selection = False

#
const_state_change = 'state_change'
const_complex_form = 'complex'
kernel_joint_trained_model_obj = ie.KernelClassifier(is_joint=True, is_protein_state=False)
kernel_protein_state_trained_model_obj = ie.KernelClassifier(is_protein_state=True, is_joint=None)
#
if ce.is_protein_path_lkl_screening:
    kernel_non_joint_trained_model_obj = ie.KernelClassifier(is_joint=False)
else:
    kernel_non_joint_trained_model_obj = None
#
# filtering
is_filter = True
const_min_threshold_interaction__kernel_inferred_lkl = 0.5
const_min_threshold_protein_state_lkl = 0.5
#
if kect.top_ratio is not None:
    top_ratio = kect.top_ratio
else:
    # tuned value should be some where between 0.25 - 0.35
    if ch.is_hpcc:
        top_ratio = 1
    else:
        top_ratio = 1
print 'selected top ratio is {}'.format(top_ratio)


def gen_data_features_fr_pathway_modeling(amr_dot_file, start_amr=None, end_amr=None, protein_name_idlist_map=None):
    def get_most_lkl_state_fr_protein_concept(protein_concept_most_lkl_state_map, protein_node, concept_node):
        protein_concept_id_tuple = (protein_node.id, concept_node.id)
        if protein_concept_id_tuple not in protein_concept_most_lkl_state_map:
            amr_paths_map, tuples_map, sentences_map = \
                ead.gen_concept_state_paths_frm_amr_protein_nd_wr_dot_pdf(amr_dot_file_path, org_concept_node=concept_node,
                                                                          org_protein_node=protein_node,
                                                                          kernel_non_joint_trained_model_obj=kernel_non_joint_trained_model_obj)
            max_lkl_fr_state = 0
            max_lkl_state_str = None
            max_lkl_state_node = None
            for curr_amr_path_key in amr_paths_map:
                curr_path_nodes_list = amr_paths_map[curr_amr_path_key]
                curr_tuple = tuples_map[curr_amr_path_key]
                # not performing duplicate elimination or pruning of non-path nodes or centralization
                # assuming that all of these are done already (centralization is by default there for protein state extraction)
                curr_amr_path_map = {'path': curr_amr_path_key, 'nodes': te.get_map_frm_list(curr_path_nodes_list), 'tuple': curr_tuple}
                if not is_random_state_selection:
                    curr_protein_state_infer_vec = kernel_protein_state_trained_model_obj.infer_frm_svm_saved_classifier(test_amr_graph=curr_amr_path_map)
                    print 'curr_protein_state_infer_vec', curr_protein_state_infer_vec
                else:
                    curr_protein_state_infer_vec = nprnd.dirichlet(np.array([1, 1, 1]))
                    print 'randomly sampled curr_protein_state_infer_vec is ', curr_protein_state_infer_vec
                if curr_protein_state_infer_vec is None:
                    continue
                if curr_protein_state_infer_vec[0] > (1-const_min_threshold_protein_state_lkl):
                    continue
                if curr_protein_state_infer_vec[1] > max_lkl_fr_state:
                    max_lkl_fr_state = curr_protein_state_infer_vec[1]
                    max_lkl_state_str = curr_tuple[2].get_name_formatted()
                    max_lkl_state_node = curr_tuple[2]
                if curr_protein_state_infer_vec[2] > max_lkl_fr_state:
                    max_lkl_fr_state = curr_protein_state_infer_vec[2]
                    max_lkl_state_str = curr_tuple[1].get_name_formatted()
                    max_lkl_state_node = curr_tuple[1]
            most_lkl_state_tuple = None
            if max_lkl_state_str is not None:
                if max_lkl_state_node is None:
                    raise AssertionError
                most_lkl_state_tuple = (max_lkl_state_str, max_lkl_state_node, max_lkl_fr_state)
            protein_concept_most_lkl_state_map[protein_concept_id_tuple] = most_lkl_state_tuple
        return protein_concept_most_lkl_state_map[protein_concept_id_tuple]

    def add_state_of_proteins_in_interactions(selected_unique_interactions_map):
        protein_concept_most_lkl_state_map = {}
        for curr_interaction_key in selected_unique_interactions_map:
            curr_tuple = selected_unique_interactions_map[curr_interaction_key]['nodes_tuple']
            amr_dot_file_path = selected_unique_interactions_map[curr_interaction_key]['amr']
            org_amr_file_path = fpe.extract_original_amr_dot_file_name(amr_dot_file_path)
            print 'org_amr_file_path', org_amr_file_path
            assert org_amr_file_path is not None
            if org_amr_file_path not in protein_concept_most_lkl_state_map:
                protein_concept_most_lkl_state_map[org_amr_file_path] = {}
            # str should be from original nodes
            concept_str = curr_tuple[0].name
            concept_node = curr_tuple[0]
            assert concept_node is not None
            #
            catalyst_node = curr_tuple[1]
            if curr_interaction_key[1] is not None:
                catalyst_str = curr_tuple[1].name
            else:
                catalyst_str = None
            protein_str = curr_tuple[2].name
            protein_node = curr_tuple[2]
            assert protein_node is not None
            if len(curr_interaction_key) == 4:
                protein2_str = curr_tuple[3].name
                protein2_node = curr_tuple[3]
                assert protein2_node is not None
            # getting most likely state candidate for catalyst
            if catalyst_node is not None:
                most_lkl_catalyst_state_tuple = get_most_lkl_state_fr_protein_concept(
                    protein_concept_most_lkl_state_map[org_amr_file_path], catalyst_node, concept_node)
            # getting most likely state candidate for protein
            most_lkl_protein_state_tuple = get_most_lkl_state_fr_protein_concept(
                protein_concept_most_lkl_state_map[org_amr_file_path], protein_node, concept_node)
            # getting most likely state candidate for protein2
            if len(curr_interaction_key) == 4:
                most_lkl_protein2_state_tuple = get_most_lkl_state_fr_protein_concept(
                    protein_concept_most_lkl_state_map[org_amr_file_path], protein2_node, concept_node)
            #
            all_proteins_state_map_tuple = [None, None]
            if len(curr_interaction_key) == 4:
                all_proteins_state_map_tuple.append(None)
            if catalyst_str is not None and most_lkl_catalyst_state_tuple is not None:
                all_proteins_state_map_tuple[0] = {most_lkl_catalyst_state_tuple[0]: most_lkl_catalyst_state_tuple[1]}
            if most_lkl_protein_state_tuple is not None:
                all_proteins_state_map_tuple[1] = {most_lkl_protein_state_tuple[0]: most_lkl_protein_state_tuple[1]}
            if len(curr_interaction_key) == 4 and most_lkl_protein2_state_tuple is not None:
                all_proteins_state_map_tuple[2] = {most_lkl_protein2_state_tuple[0]: most_lkl_protein2_state_tuple[1]}
            selected_unique_interactions_map[curr_interaction_key]['all_proteins_state_map_tuple'] = all_proteins_state_map_tuple

    def select_of_extracted_interactions_frm_amrs(paths_map, interaction_tuples_map, sentences_map):
        selected_unique_interactions_map = {}
        paths_map_keys = paths_map.keys()
        paths_map_keys.sort()
        for curr_graph_key in paths_map_keys:
            print 'curr_graph_key ', curr_graph_key
            curr_nodes_list = paths_map[curr_graph_key]
            curr_nodes_list = ead.remove_duplicates_based_on_ids(curr_nodes_list)
            curr_nodes_list = ead.prune_non_path_nodes_references_frm_subgraph(curr_nodes_list)
            #
            curr_tuple = interaction_tuples_map[curr_graph_key]
            if not (3 <= len(curr_tuple) <= 4):
                raise AssertionError
            if len(curr_tuple) == 3:
                curr_interaction_key_tuple = [None, None, None]
            elif len(curr_tuple) == 4:
                curr_interaction_key_tuple = [None, None, None, None]
            curr_interaction_key_tuple[0] = curr_tuple[0].get_name_formatted().lower()
            # catalyst can be none
            if curr_tuple[1] is not None:
                curr_interaction_key_tuple[1] = curr_tuple[1].get_name_formatted().upper()
            curr_interaction_key_tuple[2] = curr_tuple[2].get_name_formatted().upper()
            if len(curr_interaction_key_tuple) == 4:
                curr_interaction_key_tuple[3] = curr_tuple[3].get_name_formatted().upper()
            curr_interaction_key_tuple = tuple(curr_interaction_key_tuple)
            print 'curr_interaction_key_tuple', curr_interaction_key_tuple
            print 'sentences_map[curr_graph_key]', sentences_map[curr_graph_key]
            if len(curr_interaction_key_tuple) != len(set(curr_interaction_key_tuple)):
                continue
            #
            if (not te.is_amr_cyclic(curr_nodes_list) and not ck.is_neighbor_kernel) or (not te.is_amr_cyclic_undirected(curr_nodes_list) and ck.is_neighbor_kernel):
                if ck.is_inverse_centralize_amr:
                    if debug:
                        print 'performing concept centralization on ', curr_graph_key
                    curr_nodes_list = ead.centralize_amr_at_root_node(curr_nodes_list)
                    if not ch.is_hpcc:
                        ead.nodes_to_dot(curr_nodes_list, curr_graph_key+'_cc')
                curr_amr_graph_map = {'path': curr_graph_key, 'nodes': te.get_map_frm_list(curr_nodes_list), 'tuple': curr_tuple}
                #infer here
                curr_infer_vec = kernel_joint_trained_model_obj.infer_frm_svm_saved_classifier(test_amr_graph=curr_amr_graph_map)
                print 'curr_infer_vec ', curr_infer_vec
                #
                add_interaction_to_selections(curr_interaction_key_tuple, curr_tuple, selected_unique_interactions_map, curr_infer_vec[1],
                                              curr_nodes_list, sentences_map, curr_graph_key)
                if curr_interaction_key_tuple[1] is not None: #catalyst is not none
                    #generate noisy interactions by roles swappings
                    #swap catalyst and protein
                    if curr_interaction_key_tuple[1] != curr_interaction_key_tuple[2]:
                        curr_interaction_key_tuple_scp1, curr_tuple_scp1 = gen_new_interaction_tuple_wd_swap(curr_interaction_key_tuple, curr_tuple, 1, 2)
                        add_interaction_to_selections(curr_interaction_key_tuple_scp1, curr_tuple_scp1, selected_unique_interactions_map, curr_infer_vec[2], curr_nodes_list, sentences_map, curr_graph_key)
                    if len(curr_interaction_key_tuple) == 4:
                        if curr_interaction_key_tuple[1] != curr_interaction_key_tuple[3]:
                            curr_interaction_key_tuple_scp2, curr_tuple_scp2 = gen_new_interaction_tuple_wd_swap(curr_interaction_key_tuple, curr_tuple, 1, 3)
                            add_interaction_to_selections(curr_interaction_key_tuple_scp2, curr_tuple_scp2, selected_unique_interactions_map, curr_infer_vec[2], curr_nodes_list, sentences_map, curr_graph_key)
            else:
                print 'Graph {} is cyclic'.format(curr_graph_key)
        return selected_unique_interactions_map

    def filter_out_low_lkl_frm_selected_interactions_map(selected_unique_interactions_map):
        print 'in function filter_out_low_lkl_frm_selected_interactions_map ***********************'
        #
        lkl_values = []
        for curr_interaction in selected_unique_interactions_map.values():
            lkl_values.append(curr_interaction['lkl'])
        lkl_values.sort(reverse=True)
        print 'lkl_values', lkl_values
        if lkl_values:
            min_idx = int(math.ceil(len(lkl_values)*top_ratio))-1
            min_lkl_value = max(const_min_threshold_interaction__kernel_inferred_lkl, lkl_values[min_idx])
            lkl_values = None
        else:
            min_lkl_value = const_min_threshold_interaction__kernel_inferred_lkl
        print 'min_lkl_value', min_lkl_value
        #
        keys_list = copy.copy(selected_unique_interactions_map.keys())
        print 'keys_list', keys_list
        for curr_interaction_key_tuple in keys_list:
            # filter our interaction with too low likelihood
            print 'lkl of selected_unique_interactions_map[curr_interaction_key_tuple]', selected_unique_interactions_map[curr_interaction_key_tuple]['lkl']
            if selected_unique_interactions_map[curr_interaction_key_tuple]['lkl'] < min_lkl_value:
                print 'removing this interaction since likelihood is less than minimal threshold ', min_lkl_value
                selected_unique_interactions_map.pop(curr_interaction_key_tuple, None)
        print 'selected_unique_interactions_map.keys()', selected_unique_interactions_map.keys()
        print '***************************************************************'

    def get_interaction_objs(selected_unique_interactions_map):
        def get_state_str_list_fr_protein(state_str_node_map):
            curr_protein_state = []
            if state_str_node_map is not None:
                for curr_state_str_key in state_str_node_map:
                    curr_state_identifier = state_str_node_map[curr_state_str_key].identifier
                    curr_state = curr_state_str_key
                    if curr_state_identifier is not None and curr_state_identifier:
                        curr_state += '['+curr_state_identifier+']'
                    curr_protein_state.append(curr_state)
            return curr_protein_state

        interaction_objs_map = {}
        interaction_objs_map[const_state_change] = []
        interaction_objs_map[const_complex_form] = []
        for curr_interaction_key_tuple in selected_unique_interactions_map:
            curr_nodes_tuple = selected_unique_interactions_map[curr_interaction_key_tuple]['nodes_tuple']
            #catalyst_state
            if curr_interaction_key_tuple[1] is not None:
                curr_catalyst_state = get_state_str_list_fr_protein(
                    selected_unique_interactions_map[curr_interaction_key_tuple]['all_proteins_state_map_tuple'][0])
            else:
                curr_catalyst_state = []
            # protein state
            curr_protein_state = get_state_str_list_fr_protein(
                selected_unique_interactions_map[curr_interaction_key_tuple]['all_proteins_state_map_tuple'][1])
            if len(curr_interaction_key_tuple) == 4:
                curr_protein2_state = get_state_str_list_fr_protein(
                    selected_unique_interactions_map[curr_interaction_key_tuple]['all_proteins_state_map_tuple'][2])
            #
            if len(curr_interaction_key_tuple) == 3:
                new_state = curr_interaction_key_tuple[0]
                if curr_nodes_tuple[0].identifier is not None and curr_nodes_tuple[0].identifier:
                    new_state += '[' + curr_nodes_tuple[0].identifier + ']'
                curr_protein_result_state = curr_protein_state + [new_state]
                #
                curr_state_change_obj = it.Interaction(
                    curr_interaction_key_tuple[2], curr_protein_state, curr_interaction_key_tuple[1], curr_catalyst_state,
                    curr_protein_result_state, curr_interaction_key_tuple[0],
                    weight=selected_unique_interactions_map[curr_interaction_key_tuple]['lkl'])
                curr_state_change_obj.text_sentence = selected_unique_interactions_map[curr_interaction_key_tuple]['text']
                #
                #protein type and id
                curr_protein_node = curr_nodes_tuple[2]
                curr_state_change_obj.protein_type = curr_protein_node.type
                curr_state_change_obj.protein_id = curr_protein_node.identifier
                curr_state_change_obj.protein_id_prob = curr_protein_node.identifier_prob
                # catalyst type and id
                curr_catalyst_node = curr_nodes_tuple[1]
                if curr_catalyst_node is not None:
                    curr_state_change_obj.catalyst_type = curr_catalyst_node.type
                    curr_state_change_obj.catalyst_id = curr_catalyst_node.identifier
                    curr_state_change_obj.catalyst_id_prob = curr_catalyst_node.identifier_prob
                else:
                    if curr_interaction_key_tuple[1] is not None:
                        raise AssertionError
                #
                interaction_objs_map[const_state_change].append(curr_state_change_obj)
            elif len(curr_interaction_key_tuple) == 4:
                curr_complex_form_obj = it.ComplexTypeInteraction(
                    curr_interaction_key_tuple[2], curr_protein_state, curr_interaction_key_tuple[3], curr_protein2_state,
                    curr_interaction_key_tuple[1], curr_catalyst_state, None, None,
                    weight=selected_unique_interactions_map[curr_interaction_key_tuple]['lkl'],
                    complex_interaction_str=curr_interaction_key_tuple[0])
                curr_complex_form_obj.text_sentence = selected_unique_interactions_map[curr_interaction_key_tuple]['text']
                #
                # protein1 type and id
                curr_protein1_node = curr_nodes_tuple[2]
                curr_complex_form_obj.protein_1_type = curr_protein1_node.type
                curr_complex_form_obj.protein_1_id = curr_protein1_node.identifier
                curr_complex_form_obj.protein_1_id_prob = curr_protein1_node.identifier_prob
                # protein2 type and id
                curr_protein2_node = curr_nodes_tuple[3]
                curr_complex_form_obj.protein_2_type = curr_protein2_node.type
                curr_complex_form_obj.protein_2_id = curr_protein2_node.identifier
                curr_complex_form_obj.protein_2_id_prob = curr_protein2_node.identifier_prob
                # catalyst type and id
                curr_catalyst_node = curr_nodes_tuple[1]
                if curr_catalyst_node is not None:
                    curr_complex_form_obj.catalyst_type = curr_catalyst_node.type
                    curr_complex_form_obj.catalyst_id = curr_catalyst_node.identifier
                    curr_complex_form_obj.catalyst_id_prob = curr_catalyst_node.identifier_prob
                else:
                    if curr_interaction_key_tuple[1] is not None:
                        raise AssertionError
                #
                interaction_objs_map[const_complex_form].append(curr_complex_form_obj)
            else:
                raise AssertionError
        return interaction_objs_map

    def add_interaction_to_selections(curr_interaction_key_tuple, curr_tuple, selected_unique_interactions_map, prob_fr_valid_or_noisy_interaction, curr_nodes_list, sentences_map, curr_graph_key):
        if curr_interaction_key_tuple in selected_unique_interactions_map:
            #todo: if probability for this to be valid is more, then update the probability
            #todo: and also correspondingly get state intergration category as per this data sample (data different even though tuples may match)
            if prob_fr_valid_or_noisy_interaction > selected_unique_interactions_map[curr_interaction_key_tuple]['lkl']:
                selected_unique_interactions_map[curr_interaction_key_tuple]['lkl'] = prob_fr_valid_or_noisy_interaction
                selected_unique_interactions_map[curr_interaction_key_tuple]['nodes'] = curr_nodes_list
                selected_unique_interactions_map[curr_interaction_key_tuple]['text'] = sentences_map[curr_graph_key]
                selected_unique_interactions_map[curr_interaction_key_tuple]['amr'] = curr_graph_key
                selected_unique_interactions_map[curr_interaction_key_tuple]['nodes_tuple'] = curr_tuple
        else:
            selected_unique_interactions_map[curr_interaction_key_tuple] = {}
            selected_unique_interactions_map[curr_interaction_key_tuple]['lkl'] = prob_fr_valid_or_noisy_interaction
            selected_unique_interactions_map[curr_interaction_key_tuple]['nodes'] = curr_nodes_list
            selected_unique_interactions_map[curr_interaction_key_tuple]['text'] = sentences_map[curr_graph_key]
            selected_unique_interactions_map[curr_interaction_key_tuple]['amr'] = curr_graph_key
            selected_unique_interactions_map[curr_interaction_key_tuple]['nodes_tuple'] = curr_tuple

    def gen_new_interaction_tuple_wd_swap(curr_interaction_key_tuple, curr_tuple, catalyst_idx, protein_idx):
        curr_interaction_key_tuple_scp = copy.copy(list(curr_interaction_key_tuple))
        curr_tuple_scp = copy.copy(list(curr_tuple))
        curr_interaction_key_tuple = None
        curr_tuple = None
        #
        temp = curr_interaction_key_tuple_scp[catalyst_idx]
        curr_interaction_key_tuple_scp[catalyst_idx] = curr_interaction_key_tuple_scp[protein_idx]
        curr_interaction_key_tuple_scp[protein_idx] = temp
        temp = None
        curr_interaction_key_tuple_scp = tuple(curr_interaction_key_tuple_scp)
        #
        temp = curr_tuple_scp[catalyst_idx]
        curr_tuple_scp[catalyst_idx] = curr_tuple_scp[protein_idx]
        curr_tuple_scp[protein_idx] = temp
        temp = None
        curr_tuple_scp = tuple(curr_tuple_scp)
        #
        return curr_interaction_key_tuple_scp, curr_tuple_scp

    # main function starts here
    paths_map, interaction_tuples_map, sentences_map = gefd.gen_concept_domain_catalyst_data_features(
        amr_dot_file=amr_dot_file, start_amr=start_amr, end_amr=end_amr,
        protein_name_idlist_map=protein_name_idlist_map, kernel_non_joint_trained_model_obj=kernel_non_joint_trained_model_obj)
    #
    # todo: also infer interaction category
    #
    # todo: also infer "is_negative_information" and have a field in the pathway modeling format (no change on observation model though)
    #
    # todo: instead of representing complex formation interaction as binding, we need to use the actual concept keyword (binding interaction
    # todo: will be actually a more general interaction and not just complex formation)
    # todo: make this change in pathway modeling
    #
    selected_unique_interactions_map = select_of_extracted_interactions_frm_amrs(paths_map, interaction_tuples_map, sentences_map)
    #
    if start_amr is not None and end_amr is not None:
        amr_dot_file_path = amr_dot_file + '_' + str(start_amr) + '_' + str(end_amr) + '_interactions_kernel'
    else:
        amr_dot_file_path = amr_dot_file
    if not ch.is_hpcc:
        gen_excel_file_fr_interactions(selected_unique_interactions_map, amr_dot_file_path)
    #
    if is_filter:
        filter_out_low_lkl_frm_selected_interactions_map(selected_unique_interactions_map)
    #
    # add state information for protein in interactions here
    add_state_of_proteins_in_interactions(selected_unique_interactions_map)
    #
    interaction_objs_map = get_interaction_objs(selected_unique_interactions_map)
    #
    if not ch.is_hpcc:
        ead.print_interactions(interaction_objs_map, amr_dot_file_path+'.eif') #extracted interactions format
        ead.save_interactions(amr_dot_file_path, interaction_objs_map)
    #
    if start_amr is None and end_amr is None:
        curr_sentence = pd.graph_from_dot_file(cap.absolute_path+amr_dot_file).get_label()
        if not ch.is_hpcc:
            ead.save_sentence(amr_dot_file_path, curr_sentence)
    return interaction_objs_map


def gen_excel_file_fr_interactions(selected_unique_interactions_map, file_path):
    file_path += '.csv'
    f = open(cap.absolute_path+file_path, 'w')
    field_names = ['Category', 'Relation', 'Catalyst', 'Entity1', 'Entity2', 'Likelihood', 'Text', 'AMR']
    csv_writer = csv.DictWriter(f, fieldnames=field_names, dialect='excel', lineterminator='\n', delimiter=',')
    csv_writer.writeheader()
    for curr_interaction_tuple_key in selected_unique_interactions_map:
        curr_row_map = {'Category': None, 'Relation': curr_interaction_tuple_key[0], 'Catalyst': curr_interaction_tuple_key[1], 'Entity1': curr_interaction_tuple_key[2], 'Likelihood': selected_unique_interactions_map[curr_interaction_tuple_key]['lkl'], 'Text': selected_unique_interactions_map[curr_interaction_tuple_key]['text'], 'AMR': selected_unique_interactions_map[curr_interaction_tuple_key]['amr']}
        if len(curr_interaction_tuple_key) == 4:
            curr_row_map['Entity2'] = curr_interaction_tuple_key[3]
        csv_writer.writerow(curr_row_map)
    f.close()


if __name__ == '__main__':
    import sys
    amr_dot_file = str(sys.argv[1])
    start_amr = int(sys.argv[2])
    end_amr = int(sys.argv[3])
    gen_data_features_fr_pathway_modeling(amr_dot_file=amr_dot_file, start_amr=start_amr, end_amr=end_amr)
