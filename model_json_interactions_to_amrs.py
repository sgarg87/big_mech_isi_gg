import gen_extractor_features_data as gefd
import constants_darpa_json_format as cd
import darpa_interaction_types as dit
import constants as c
import extract_from_amr_dot as ead
import os
import shutil
import constants_absolute_path as cap
import csv
from config_console_output import *


modifications_map = {'phosphorylation': c.phosphorylate, 'acetylation': c.acetylate, 'deacetylate': c.deacetylate,
                     'farnesylation': c.farnesylate, 'glycosylate': c.glycosylate, 'hydroxylation': c.hydrolyze,
                     'methylation': c.methylate, 'ribosylation': c.ribosylate, 'sumoylation': c.sumoylate,
                     'ubiquitination': c.ubiquitinate}


class InteractionsJSONToAMR:
    def __init__(self):
        self.count_joint = 0
        self.file_path_joint = './json_darpa_model_amrs/triplet_joint_subgraphs/'
        if os.path.exists(cap.absolute_path+self.file_path_joint):
            shutil.rmtree(cap.absolute_path+self.file_path_joint)
        os.makedirs(cap.absolute_path+self.file_path_joint)
        self.f = open('./json_darpa_model_amrs/interactions.csv', 'wb')
        field_name = ['concept', 'catalyst', 'protein1', 'protein2']
        self.csv_writer = csv.DictWriter(self.f, field_name)
        self.csv_writer.writeheader()

    def __del__(self):
        self.f.close()

    def get_sentence(self, curr_json_obj):
        if cd.evidence in curr_json_obj and curr_json_obj[cd.evidence]:
            if len(curr_json_obj[cd.evidence]) > 1:
                raise NotImplementedError
            if curr_json_obj[cd.evidence][0]:
                sentence = curr_json_obj[cd.evidence][0]
            else:
                sentence = 'no text'
        else:
            sentence = 'no text'
        return sentence

    def add_subgraph_from_state_change_interaction(self, curr_json_obj, data):
        is_negative_polarity = False
        catalyst_str = None
        if curr_json_obj[cd.extracted_information][cd.participant_a] is not None:
            if cd.entities in curr_json_obj[cd.extracted_information][cd.participant_a]:
                raise NotImplementedError
            catalyst_str = curr_json_obj[cd.extracted_information][cd.participant_a][cd.entity_text]
        if cd.entities in curr_json_obj[cd.extracted_information][cd.participant_b]:
            raise NotImplementedError
        protein_str = curr_json_obj[cd.extracted_information][cd.participant_b][cd.entity_text]
        concept_terms = []
        if curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases:
            concept_terms.append(c.decrease)
            concept_terms.append(c.degrade)
        elif curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases:
            concept_terms.append(c.increase)
            concept_terms.append(c.express)
            concept_terms.append(c.transcribe)
            concept_terms.append(c.produce)
            # concept_terms.append(c.enhance)
        elif curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.increases_activity:
            concept_terms.append(c.activate)
            # concept_terms.append(c.stimulate)
            if catalyst_str is None:
                concept_terms.append(c.signal)
                concept_terms.append(c.act)
        elif curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.decreases_activity:
            concept_terms.append(c.inhibit)
            concept_terms.append(c.impede)
            # concept_terms.append(c.supress)
            concept_terms.append(c.diminish)
        elif curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.translocates:
            concept_terms.append(c.recruit)
            concept_terms.append(c.translocate)
            concept_terms.append(c.localize)
            # concept_terms.append(c.relocalize)
        elif curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.adds_modification:
            for curr_modification in curr_json_obj[cd.extracted_information][cd.modification]:
                print 'curr_modification', curr_modification
                concept_terms.append(modifications_map[curr_modification[cd.modification_type]])
        elif curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.removes_modification:
            is_negative_polarity = True
            for curr_modification in curr_json_obj[cd.extracted_information][cd.modification]:
                print 'curr_modification', curr_modification
                concept_terms.append(modifications_map[curr_modification[cd.modification_type]])
        concept_terms = list(set(concept_terms))
        print 'concept_terms', concept_terms
        for curr_concept_term in concept_terms:
            curr_amr_info_map = self.gen_amr_triplet_joint(curr_concept_term, catalyst_str, protein_str,
                                                           is_negative_polarity=is_negative_polarity)
            self.add_interaction_to_csv(curr_concept_term, catalyst_str, protein_str)
            sentence = self.get_sentence(curr_json_obj)
            self.count_joint += 1
            path_key = self.file_path_joint+'state_change_'+str(self.count_joint)
            ead.nodes_to_dot(curr_amr_info_map['nodes'], path_key, sentence)
            #
            data[gefd.const_paths_map][path_key] = curr_amr_info_map['nodes']
            data[gefd.const_interaction_tuples_map][path_key] = curr_amr_info_map['tuple']
            data[gefd.const_sentences_map][path_key] = sentence
            data[gefd.const_joint_labels_map][path_key] = 1

    def add_subgraph_from_complex_form_type_interaction(self, curr_json_obj, data):
        is_negative_polarity = False
        concept_terms = [c.bind, c.associate]
        if curr_json_obj[cd.extracted_information][cd.interaction_type] in [dit.increases, dit.decreases]:
            if dit.decreases:
                is_negative_polarity = True
            catalyst_str = curr_json_obj[cd.extracted_information][cd.participant_a][cd.entity_text]
            protein1_str = curr_json_obj[cd.extracted_information][cd.participant_b][cd.entities][0][cd.entity_text]
            protein2_str = curr_json_obj[cd.extracted_information][cd.participant_b][cd.entities][1][cd.entity_text]
        elif curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.binds:
            catalyst_str = None
            protein1_str = curr_json_obj[cd.extracted_information][cd.participant_a][cd.entity_text]
            protein2_str = curr_json_obj[cd.extracted_information][cd.participant_b][cd.entity_text]
        else:
            raise AssertionError
        for curr_concept_term in concept_terms:
            curr_amr_info_map = self.gen_amr_triplet_joint(curr_concept_term, catalyst_str, protein1_str, protein2_str,
                                                           is_negative_polarity=is_negative_polarity)
            self.add_interaction_to_csv(curr_concept_term, catalyst_str, protein1_str, protein2_str)
            sentence = self.get_sentence(curr_json_obj)
            self.count_joint += 1
            path_key = self.file_path_joint+'complex_form_type_'+str(self.count_joint)
            ead.nodes_to_dot(curr_amr_info_map['nodes'], path_key, sentence)
            #
            data[gefd.const_paths_map][path_key] = curr_amr_info_map['nodes']
            data[gefd.const_interaction_tuples_map][path_key] = curr_amr_info_map['tuple']
            data[gefd.const_sentences_map][path_key] = sentence
            data[gefd.const_joint_labels_map][path_key] = 1

    def gen_amr_triplet_joint(self, org_concept_str, catalyst_str, protein_str, protein2_str=None, is_negative_polarity=False, label=1):
        curr_nodes_list = []
        #
        assert org_concept_str is not None and org_concept_str
        concept_str = org_concept_str + '-99'
        concept_node = ead.Node(id='c1', name=concept_str)
        concept_node.color = 'blue'
        concept_str = None
        curr_nodes_list.append(concept_node)
        #
        if catalyst_str is not None and catalyst_str:
            catalyst_node = ead.Node(id='c2', name=catalyst_str)
            catalyst_node.type = 'protein'
            catalyst_node.color = 'green'
            catalyst_str = None
        else:
            catalyst_node = None
        if catalyst_node is not None:
            curr_nodes_list.append(catalyst_node)
        #
        assert protein_str is not None and protein_str
        protein_node = ead.Node(id='p1', name=protein_str)
        protein_node.type = 'protein'
        protein_node.color = '#976850'
        curr_nodes_list.append(protein_node)
        #
        if protein2_str is not None and protein2_str:
            protein2_node = ead.Node(id='p2', name=protein2_str)
            protein2_node.type = 'protein'
            protein2_node.color = '#976856'
        else:
            protein2_node = None
        if protein2_node is not None:
            curr_nodes_list.append(protein2_node)
        #
        curr_amr_map = {}
        #
        curr_triplet_tuple = [concept_node, catalyst_node, protein_node]
        if protein2_node is not None:
            curr_triplet_tuple.append(protein2_node)
        #
        if label == 1:
            arg0 = 'ARG0'
            arg1 = 'ARG1'
            arg2 = 'ARG2'
        elif label == 2:
            if catalyst_node is None:
                raise AssertionError
            arg0 = 'ARG1'
            arg1 = 'ARG0'
            arg2 = 'ARG2'
        elif label == 0:
            if catalyst_node is None:
                arg1 = 'ARG0'
                arg0 = 'ARG1'
                arg2 = 'ARG2'
            else:
                arg0 = 'ARG4'
                arg1 = 'ARG5'
                arg2 = 'ARG6'
        else:
            raise AssertionError
        #
        if org_concept_str in [c.signal, c.act]:
            assert catalyst_node is None
            concept_node.add_parent_child_relationship(protein_node, arg0)
        else:
            concept_node.add_parent_child_relationship(protein_node, arg1)
        #
        if catalyst_node is not None:
            concept_node.add_parent_child_relationship(catalyst_node, arg0)
        if protein2_node is not None:
            concept_node.add_parent_child_relationship(protein2_node, arg2)
        #
        if is_negative_polarity:
            polarity_node = ead.Node(id='n1', name='-')
            curr_nodes_list.append(polarity_node)
            concept_node.add_parent_child_relationship(polarity_node, 'polarity')
        #
        curr_amr_map['tuple'] = tuple(curr_triplet_tuple)
        curr_amr_map['nodes'] = curr_nodes_list
        #
        return curr_amr_map

    def add_interaction_to_csv(self, concept_str, catalyst_str, protein_str, protein2_str=None):
        self.csv_writer.writerow({'concept': concept_str, 'catalyst': catalyst_str, 'protein1': protein_str, 'protein2': protein2_str})

    def gen_amr_subgraphs_frm_model_interactions(self, json_objs_list, is_pickle=False):
        print 'No. of input json objs is ', len(json_objs_list)
        # # temporary
        # temp_json_list = []
        # for curr_json_obj in json_objs_list:
        #     if curr_json_obj not in temp_json_list:
        #         temp_json_list.append(curr_json_obj)
        # print 'No of unique json objs is ', len(temp_json_list)
        # temp_json_list = None
        # #
        # temp_json_str_list = []
        # for curr_json_obj in json_objs_list:
        #     if str(curr_json_obj) not in temp_json_str_list:
        #         temp_json_str_list.append(str(curr_json_obj))
        # print 'No of unique json str of objs is ', len(temp_json_str_list)
        # temp_json_str_list = None
        #
        data = {}
        data[gefd.const_paths_map] = {}
        data[gefd.const_interaction_tuples_map] = {}
        data[gefd.const_sentences_map] = {}
        data[gefd.const_joint_labels_map] = {}
        #
        for curr_json_obj in json_objs_list:
            if curr_json_obj[cd.extracted_information][cd.interaction_type] == dit.binds\
                    or ((curr_json_obj[cd.extracted_information][cd.interaction_type] in [dit.increases, dit.decreases])
                        and (cd.entities in curr_json_obj[cd.extracted_information][cd.participant_b])):
                self.add_subgraph_from_complex_form_type_interaction(curr_json_obj, data)
            else:
                self.add_subgraph_from_state_change_interaction(curr_json_obj, data)
        if is_pickle:
            gefd.dump_pickle_data_joint_model(data, is_extend=False, is_synthetic=False)
        return data


if __name__ == '__main__':
    import biopax_model_obj as bmo
    ija_obj = InteractionsJSONToAMR()
    ija_obj.gen_amr_subgraphs_frm_model_interactions(bmo.bm_obj.json_objs, is_pickle=True)

