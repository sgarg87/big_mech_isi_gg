import darpa_participant_types as dpt
import wordsegment as ws
from config import *
import model_relation_types as mrt
import darpa_interaction_types as dit
import interactions as i
import time
import json
import re
import constants_absolute_path as cap
import extract_from_amr_dot as ead
import gen_extractor_features_data as gefd
import constants_joint_synthetic as cjs
import constants_protein_state_synthetic as cpss
import constants_darpa_json_format as cdjf
import relation_to_model as rm
import os
import shutil
import config_hpcc as ch
from constants_locations_go_identifiers import *
import uniprot_mapping_obj as umo
import darpa_participant_types_mapping as dptm
from config_console_output import *
import biopax_model_obj as bmo
import config_darpa_json as cdj
import constants as c
import config_darpa as cd


is_skip = True


def get_mutation_feature_type(curr_str):
    is_mutation = False
    if 'mut' in curr_str:
        is_mutation = True
    # elif 'canc' in curr_str:
    #     is_mutation = True
    # elif 'onco' in curr_str:
    #     is_mutation = True
    # elif 'tumor' in curr_str:
    #     is_mutation = True
    elif len(re.findall(r'A\d+', curr_str)) > 0:
        is_mutation = True
    if is_mutation:
        feature_type = 'mutation'
    else:
        feature_type = None
    return feature_type


def extract_darpa_mod_features(state_str, is_feature_type=True):
    # todo: this function currently gives site position and site type in a factorized manner assuming that there is only one site type in a give state str
    def get_site_type(curr_str):
        curr_str = curr_str.lower()
        site_type = None
        if 'ser' in curr_str:
            site_type = 'serine'
        elif 'tyr' in curr_str:
            site_type = 'tyrosine'
        elif 'thr' in curr_str:
            site_type = 'threonine'
        return site_type

    def get_mod_type(curr_str):
        if 'phosph' in state_str:
            mod_type = 'phosphorylation'
        elif 'ubiquit' in state_str:
            mod_type = 'ubiquitination'
        elif 'acety' in state_str:
            mod_type = 'acetylation'
        elif 'farnes' in state_str:
            mod_type = 'farnesylation'
        elif 'glycos' in state_str:
            mod_type = 'glycosylation'
        elif 'hydro' in state_str:
            mod_type = 'hydroxylation'
        elif 'meth' in state_str:
            mod_type = 'methylation'
        elif 'ribos' in state_str:
            mod_type = 'ribosylation'
        elif 'sumo' in state_str:
            mod_type = 'sumoylation'
        elif 'carbox' and 'meth' in state_str:
            mod_type = 'carboxymethylation'
        elif 'depalm' in state_str:
            mod_type = 'depalmitoylation'
        else:
            mod_type = None
        return mod_type

    def get_site_positions(curr_str):
        site_number_list = []
        site_number_str_range = re.findall(r'\d+-\d+', curr_str)
        if len(site_number_str_range) > 1:
            raise NotImplementedError
        elif len(site_number_str_range) == 1:
            site_number_str_range_start = int(re.findall(r'\d+-', site_number_str_range[0])[0].strip('-'))
            site_number_str_range_end = int(re.findall(r'-\d+', site_number_str_range[0])[0].strip('-'))
            site_number_list = range(site_number_str_range_start, site_number_str_range_end+1)
        curr_site_numbers = re.findall(r'\d+', curr_str)
        print 'curr_site_numbers', curr_site_numbers
        for curr_site_number in curr_site_numbers:
            curr_site_number = int(curr_site_number)
            if curr_site_number not in site_number_list:
                site_number_list.append(curr_site_number)
        print 'site_number_list', site_number_list
        if site_number_list:
            return site_number_list

    if state_str is None or not state_str:
        return None, None, None
    state_str = state_str.lower()
    mod_type = get_mod_type(state_str)
    if is_feature_type:
        if mod_type is None or not mod_type:
            mod_type = get_mutation_feature_type(state_str)
    site_type = get_site_type(state_str)
    site_positions_list = get_site_positions(state_str)
    type_pos_map = None
    if (site_type is not None and site_type) or (site_positions_list is not None and site_positions_list):
        type_pos_map = {}
        if site_type is not None:
            type_pos_map['site_type'] = site_type
        if site_positions_list:
            type_pos_map['pos'] = site_positions_list
    # print 'type_pos_map', type_pos_map
    return mod_type, type_pos_map


def is_activity_type(curr_state):
    is_curr_act_type = False
    if 'act' in curr_state:
        is_curr_act_type = True
    elif 'sign' in curr_state:
        is_curr_act_type = True
    elif 'stim' in curr_state:
        is_curr_act_type = True
    elif 'inhib' in curr_state:
        is_curr_act_type = True
    elif 'imped' in curr_state:
        is_curr_act_type = True
    elif 'supr' in curr_state:
        is_curr_act_type = True
    elif 'dimin' in curr_state:
        is_curr_act_type = True
    elif 'GTP' in curr_state.upper():
        is_curr_act_type = True
    elif 'GDP' in curr_state.upper():
        is_curr_act_type = True
    return is_curr_act_type


def is_decrease_activity(curr_state):
    is_dec_type = False
    if 'de' in curr_state:
        is_dec_type = True
    if 'inhib' in curr_state:
        is_dec_type = True
    elif 'imped' in curr_state:
        is_dec_type = True
    elif 'supr' in curr_state:
        is_dec_type = True
    elif 'dimin' in curr_state:
        is_dec_type = True
    elif 'GDP' in curr_state.upper():
        is_dec_type = True
    return is_dec_type


def is_decrease_type(curr_state):
    is_dec_type = False
    if 'decre' in curr_state:
        is_dec_type = True
    elif 'degrad' in curr_state:
        is_dec_type = True
    return is_dec_type


def is_increase_type(curr_state):
    is_inc_type = False
    if 'incre' in curr_state:
        is_inc_type = True
    elif 'expr' in curr_state:
        is_inc_type = True
    elif 'transc' in curr_state:
        is_inc_type = True
    elif 'prod' in curr_state:
        is_inc_type = True
    elif 'potent' in curr_state:
        is_inc_type = True
    elif 'enhan' in curr_state:
        is_inc_type = True
    return is_inc_type


def remove_translocate_type_term_frm_list(curr_list):
    new_list = []
    for curr_term in curr_list:
        if 'loca' in curr_term.lower() or 'translo' in curr_term.lower() or 'recru' in curr_term.lower():
            continue
        new_list.append(curr_term)
    return new_list


def get_features_map(state_str_list, is_feature_type=True):
    darpa_mod_features_map = {}
    other_features_list = []
    site_maps_list = []
    if state_str_list is None or not state_str_list:
        return darpa_mod_features_map, other_features_list
    for curr_state_str in state_str_list:
        if curr_state_str is None:
            raise AssertionError
        if not curr_state_str:
            continue
        darpa_mod_type, site_type_positions_map = extract_darpa_mod_features(curr_state_str, is_feature_type)
        if darpa_mod_type is not None:
            if darpa_mod_type not in darpa_mod_features_map:
                darpa_mod_features_map[darpa_mod_type] = []
            if site_type_positions_map is not None:
                darpa_mod_features_map[darpa_mod_type].append(site_type_positions_map)
        elif site_type_positions_map is not None:
            site_maps_list.append(site_type_positions_map)
        # todo: also consider mutation and binding
        else:
            other_features_list.append(curr_state_str)
    for curr_key in darpa_mod_features_map:
        for curr_site_type_positions_map in site_maps_list:
            darpa_mod_features_map[curr_key].append(curr_site_type_positions_map)
    return darpa_mod_features_map, other_features_list


def extract_GO_identifier_from_loc_state_str(state_loc_str):
    print 'extracting GO identifier from state_loc_str', state_loc_str
    if 'GO' not in state_loc_str:
        return None
    m = re.findall('\[(.+?)\]', state_loc_str)
    if len(m) > 1:
        raise AssertionError
    elif len(m) == 0:
        return None
    else:
        return m[0].strip('[').strip(']')


def get_GO_identifier_fr_loc_state(state_loc_str):
    print 'getting GO identifier for state_loc_str', state_loc_str
    go_id = extract_GO_identifier_from_loc_state_str(state_loc_str)
    if go_id is not None and go_id.strip():
        return go_id
    #
    # cell, cell_junction, chromatin, chromosome, chromosomal, cytoplasm, cytoskeleton, cytosol, endoplasmic_reticulum,
    # golgi, golgi_apparatus, membrane, mitochondria, mitochondrion, nuclear_membrane, nucleoplasm, nucleus, organelle,
    # plasma_membrane, proteasome, proteasome_complex, reticulum, ribosome, spliceosomal_complex, spliceosome
    #
    state_loc_str = state_loc_str.strip()
    state_loc_str = state_loc_str.lower()
    if cell == state_loc_str:
        go_id = GO_location_id_map[cell]
    elif cell in state_loc_str and 'junc' in state_loc_str:
        go_id = GO_location_id_map[cell_junction]
    elif 'endoplasm' in state_loc_str and 'retic' in state_loc_str:
        go_id = GO_location_id_map[endoplasmic_reticulum]
    elif 'golg' in state_loc_str and 'appara' in state_loc_str:
        go_id = GO_location_id_map[golgi_apparatus]
    elif 'nucl' in state_loc_str and 'memb' in state_loc_str:
        go_id = GO_location_id_map[nuclear_membrane]
    elif 'plas' in state_loc_str and 'memb' in state_loc_str:
        go_id = GO_location_id_map[plasma_membrane]
    elif 'protea' in state_loc_str and 'compl' in state_loc_str:
        go_id = GO_location_id_map[proteasome_complex]
    elif 'splic' in state_loc_str and 'compl' in state_loc_str:
        go_id = GO_location_id_map[spliceosomal_complex]
    elif 'extr' in state_loc_str and 'cell' in state_loc_str and 'matr' in state_loc_str:
        go_id = GO_location_id_map[extra_cellular_matrix]
    else:
        if state_loc_str in GO_location_id_map:
            go_id = GO_location_id_map[state_loc_str]
        else:
            print 'could not retrieve GO for ', state_loc_str
    return go_id


def is_loc_type(curr_state):
    is_curr_loc_type = False
    if is_loc(curr_state):
        is_curr_loc_type = True
    elif 'translo' in curr_state:
        is_curr_loc_type = True
    elif 'loca' in curr_state:
        is_curr_loc_type = True
    elif 'recru' in curr_state:
        is_curr_loc_type = True
    return is_curr_loc_type


def is_loc(curr_state):
    is_loc = False
    if 'cyto' in curr_state:
        is_loc = True
    elif 'plasm' in curr_state:
        is_loc = True
    elif 'nucl' in curr_state:
        is_loc = True
    elif 'memb' in curr_state:
        is_loc = True
    elif 'extra' in curr_state:
        is_loc = True
    elif 'cell' in curr_state:
        is_loc = True
    elif 'chrom' in curr_state:
        is_loc = True
    elif 'junc' in curr_state:
        is_loc = True
    elif 'retic' in curr_state:
        is_loc = True
    elif 'golg' in curr_state:
        is_loc = True
    elif 'appar' in curr_state:
        is_loc = True
    elif 'mito' in curr_state:
        is_loc = True
    elif 'chond' in curr_state:
        is_loc = True
    elif 'orga' in curr_state:
        is_loc = True
    elif 'proteas' in curr_state:
        is_loc = True
    elif 'ribo' in curr_state:
        is_loc = True
    elif 'splic' in curr_state:
        is_loc = True
    return is_loc


def get_identifier_of_protein_name(name_str, protein_name_idlist_map, name_identifier_map_amrs=None):
    def get_id_frm_protein_name_idlist_map(curr_str):
        id = None
        if curr_str in protein_name_idlist_map:
            ids_list = protein_name_idlist_map[curr_str]
            if ids_list is not None and ids_list:
                id = ids_list[0]
        if curr_str.lower() in protein_name_idlist_map:
            ids_list = protein_name_idlist_map[curr_str.lower()]
            if ids_list is not None and ids_list:
                id = ids_list[0]
        if id is not None and 'uniprot' in id.lower():
            identifier_uniprot_mapping = umo.um_obj.get_mapping(id)
            if identifier_uniprot_mapping is not None:
                id = identifier_uniprot_mapping
        return id

    def get_id_frm_name_identifier_map_amrs(curr_str):
        if name_identifier_map_amrs is not None and name_identifier_map_amrs:
            if curr_str in name_identifier_map_amrs:
                return name_identifier_map_amrs[curr_str]
            if curr_str.lower() in name_identifier_map_amrs:
                return name_identifier_map_amrs[curr_str.lower()]

    def get_ids_map_fr_split(name_str, split_char=None):
        if name_str is not None and name_str:
            if split_char is not None:
                protein_names_str = name_str.split(split_char)
            else:
                protein_names_str = ws.segment(name_str)
            name_id_map = {}
            for curr_protein_split in protein_names_str:
                curr_split_id_frm_model = get_id_frm_protein_name_idlist_map(curr_protein_split)
                curr_split_id_frm_amrs = get_id_frm_name_identifier_map_amrs(curr_protein_split)
                if curr_split_id_frm_model is not None:
                    name_id_map[curr_protein_split] = curr_split_id_frm_model
                elif curr_split_id_frm_amrs is not None:
                    name_id_map[curr_protein_split] = curr_split_id_frm_amrs
            if name_id_map:
                return name_id_map

    if name_str is None or not name_str:
        return None
    # since a protein name can actually be a complex, a map is return with name and corresponding id
    id_frm_model = get_id_frm_protein_name_idlist_map(name_str)
    id_frm_amrs = get_id_frm_name_identifier_map_amrs(name_str)
    if id_frm_model is not None:
        return id_frm_model
    elif id_frm_amrs is not None:
        return id_frm_amrs
    else:
        split_chars_list_prioritized = [' ', ',', '/', ';', ':', '-', '_', None]
        for curr_split_char in split_chars_list_prioritized:
            ids_map = get_ids_map_fr_split(name_str, curr_split_char)
            if ids_map is not None and ids_map:
                return ','.join(ids_map.values())


class Interaction_to_JSON:

    def __init__(self, is_model=False):
        self.is_model = is_model

    def add_protein_primary_information(self, participant, protein_str, protein_type, protein_id, protein_state_str_list, protein_name_idlist_map, proteins_in_model_lower_case, name_identifier_map_amrs=None):
        participant[cdjf.entity_text] = protein_str
        curr_protein_type = dptm.get_darpa_protein_type(protein_type)
        if curr_protein_type is not None:
            participant[cdjf.entity_type] = curr_protein_type
        else:
            participant[cdjf.entity_type] = ''
        #
        if protein_id is not None and protein_id:
            participant[cdjf.identifier] = protein_id
        else:
            curr_identifier = get_identifier_of_protein_name(protein_str, protein_name_idlist_map, name_identifier_map_amrs)
            if curr_identifier is not None:
                participant[cdjf.identifier] = curr_identifier
            else:
                participant[cdjf.identifier] = ''
        #
        if protein_name_idlist_map is not None:
            if (not self.is_model) and participant[cdjf.identifier] is not None and participant[cdjf.identifier]\
                    and participant[cdjf.identifier] in bmo.bm_obj.identifiers_list:
                participant[cdjf.in_model] = True
            elif protein_str.lower() in proteins_in_model_lower_case:
                participant[cdjf.in_model] = True
            else:
                participant[cdjf.in_model] = False
                for curr_model_protein_name in proteins_in_model_lower_case:
                    if curr_model_protein_name in protein_str.lower():
                        print 'curr_model_protein_name', curr_model_protein_name
                        print 'protein_str.lower()', protein_str.lower()
                        print 'proteins_in_model_lower_case', proteins_in_model_lower_case
                        participant[cdjf.in_model] = True
                        break
        else:
            raise AssertionError
        # debugging
        if (not participant[cdjf.in_model]) and (not self.is_model):
            print 'participant[cdjf.identifier]', participant[cdjf.identifier]
            print 'participant[cdjf.in_model]', participant[cdjf.in_model]
            # print 'bmo.bm_obj.identifiers_list', bmo.bm_obj.identifiers_list
        #
        features = self.get_features_list(protein_state_str_list)
        if features is not None and features:
            participant[cdjf.features] = features
        # else:
        #     participant[cdjf.features] = []
        # participant[cdjf.not_features] = []
        return participant[cdjf.in_model]

    def all_loc_type_state(self, state_list):
        loc_type_only_state = []
        if state_list is None or not state_list:
            return False, state_list
        is_loc_type_all = True
        for curr_state in state_list:
            is_curr_loc_type = is_loc_type(curr_state)
            if not is_curr_loc_type:
                is_loc_type_all = False
            else:
                loc_type_only_state.append(curr_state)
        return is_loc_type_all, loc_type_only_state

    def all_mutation_type_state(self, state_list):
        if state_list is None or not state_list:
            return False
        is_mutation_type_all = True
        for curr_state in state_list:
            is_curr_mutation_type = False
            if get_mutation_feature_type(curr_state) is not None:
                is_curr_mutation_type = True
            elif re.findall(r'\d+', curr_state):
                is_curr_mutation_type = True
            if not is_curr_mutation_type:
                is_mutation_type_all = False
                break
        return is_mutation_type_all

    def get_features_list(self, state_str_list, is_feature_type=True):
        if state_str_list is not None and state_str_list:
            state_str_list = list(set(state_str_list))
            darpa_mod_features_map, other_features_list = get_features_map(state_str_list, is_feature_type)
            features = []
            if is_feature_type:
                for curr_mod in other_features_list:
                    curr_feature_map = {}
                    if is_loc(curr_mod):
                        curr_mod_identifier = get_GO_identifier_fr_loc_state(curr_mod)
                        if curr_mod_identifier is None:
                            curr_mod_identifier = ''
                        curr_feature_map[cdjf.location_text] = curr_mod
                        curr_feature_map[cdjf.location_id] = curr_mod_identifier
                    elif is_activity_type(curr_mod):
                        if not is_decrease_activity(curr_mod):
                            curr_feature_map[cdjf.is_active] = True
                        else:
                            curr_feature_map[cdjf.is_active] = False
                    else:
                        continue
                        curr_feature_map[cdjf.feature_type] = cdjf.general
                        curr_feature_map[cdjf.feature_value] = curr_mod
                    features.append(curr_feature_map)
            for curr_mod_type in darpa_mod_features_map:
                print 'curr_mod_type', curr_mod_type
                is_mutation = False
                if curr_mod_type == 'mutation':
                    is_mutation = True
                if is_mutation and not is_feature_type:
                    continue
                if darpa_mod_features_map[curr_mod_type]:
                    for curr_site_map in darpa_mod_features_map[curr_mod_type]:
                        print 'curr_site_map', curr_site_map
                        if 'pos' in curr_site_map:
                            for curr_pos in curr_site_map['pos']:
                                print 'curr_pos', curr_pos
                                curr_feature_map = {}
                                features.append(curr_feature_map)
                                if is_feature_type:
                                    if not is_mutation:
                                        curr_feature_map[cdjf.feature_type] = cdjf.modification
                                    else:
                                        curr_feature_map[cdjf.feature_type] = cdjf.mutation
                                if not is_mutation:
                                    curr_feature_map[cdjf.modification_type] = curr_mod_type
                                curr_feature_map[cdjf.position] = curr_pos
                                if is_mutation:
                                    curr_feature_map[cdjf.from_base] = ''
                                    curr_feature_map[cdjf.to_base] = ''
                else:
                    curr_feature_map = {}
                    features.append(curr_feature_map)
                    if is_feature_type:
                        if not is_mutation:
                            curr_feature_map[cdjf.feature_type] = cdjf.modification
                        else:
                            curr_feature_map[cdjf.feature_type] = cdjf.mutation
                    if not is_mutation:
                        curr_feature_map[cdjf.modification_type] = curr_mod_type
                    curr_feature_map[cdjf.position] = 0
                    if is_mutation:
                        curr_feature_map[cdjf.from_base] = ''
                        curr_feature_map[cdjf.to_base] = ''
            if features:
                return features

    def interaction_obj_to_darpa_json(self, interactions_list, amr_parsing_frm_txt_compute_time=None, extraction_compute_time=None, pmc_id=None, protein_name_idlist_map=None, name_identifier_map_amrs=None):
        if protein_name_idlist_map is not None:
            proteins_in_model_lower_case = protein_name_idlist_map.keys()
            proteins_in_model_lower_case = [x.lower() for x in proteins_in_model_lower_case]
        else:
            proteins_in_model_lower_case = None
        json_interaction_objs = []
        for curr_obj in interactions_list:
            is_connected_to_model = False
            if debug:
                print '***********************************'
                print 'curr_obj is ', curr_obj
                print curr_obj.other_source_id
                print curr_obj.pmc_id
                print curr_obj.text_sentence
            #
            curr_json_map = {}
            curr_json_map[cdjf.submitter] = cdjf.ISI_USC
            curr_json_map[cdjf.reader_type] = cdjf.machine
            if pmc_id is not None:
                curr_json_map[cdjf.pmc_id] = pmc_id
            else:
                curr_json_map[cdjf.pmc_id] = curr_obj.pmc_id
            curr_json_map[cdjf.evidence] = [curr_obj.text_sentence.strip('"')]
            if curr_obj.weight is not None:
                curr_json_map[cdjf.weight] = curr_obj.weight
            if curr_obj.other_source_id is not None and curr_obj.other_source_id.strip():
                curr_json_map[cdjf.biopax_id] = curr_obj.other_source_id
            #
            # if mrt.Corroboration in curr_obj.model_relation_map:
            #     curr_json_map[cdjf.model_relation] = mrt.Corroboration
            #     curr_json_map[cdjf.model_elements] = [curr_obj.model_relation_map[mrt.Corroboration].other_source_id]
            # elif mrt.Conflicting in curr_obj.model_relation_map:
            #     curr_json_map[cdjf.model_relation] = mrt.Conflicting
            #     curr_json_map[cdjf.model_elements] = [curr_obj.model_relation_map[mrt.Conflicting].other_source_id]
            # elif mrt.Specialization in curr_obj.model_relation_map:
            #     curr_json_map[cdjf.model_relation] = mrt.Specialization
            #     curr_json_map[cdjf.model_elements] = [curr_obj.model_relation_map[mrt.Specialization].other_source_id]
            # else:
            #     curr_json_map[cdjf.model_relation] = mrt.Extension
            #
            identifier_prob_list = []
            if cdj.is_identifier_prob:
                curr_json_map[cdjf.identifier_prob_list] = identifier_prob_list
            if isinstance(curr_obj, i.ComplexTypeInteraction):
                # todo: catalyst case is not modeled yet
                if curr_obj.is_left_to_right:
                    if curr_obj.catalyst_str is None:
                        #
                        extracted_information = {}
                        curr_json_map[cdjf.extracted_information] = extracted_information
                        #
                        participant_a = {}
                        extracted_information[cdjf.participant_a] = participant_a
                        is_connected_to_model = self.add_protein_primary_information(participant_a, curr_obj.protein_1_str, curr_obj.protein_1_type,
                                                        curr_obj.protein_1_id, curr_obj.protein_1_state_str_list,
                                                        protein_name_idlist_map, proteins_in_model_lower_case, name_identifier_map_amrs) or is_connected_to_model
                        if cdj.is_identifier_prob:
                            identifier_prob_list.append(curr_obj.protein_1_id_prob)
                        #
                        participant_b = {}
                        extracted_information[cdjf.participant_b] = participant_b
                        is_connected_to_model = self.add_protein_primary_information(participant_b, curr_obj.protein_2_str, curr_obj.protein_2_type,
                                                        curr_obj.protein_2_id, curr_obj.protein_2_state_str_list,
                                                        protein_name_idlist_map, proteins_in_model_lower_case, name_identifier_map_amrs) or is_connected_to_model
                        if cdj.is_identifier_prob:
                            identifier_prob_list.append(curr_obj.protein_2_id_prob)
                        #
                        extracted_information[cdjf.interaction_type] = dit.binds
                        extracted_information[cdjf.participant_a_site] = ''
                        extracted_information[cdjf.participant_b_site] = ''
                        extracted_information[cdjf.negative_information] = curr_obj.is_negative_information
                    else:
                        #
                        extracted_information = {}
                        curr_json_map[cdjf.extracted_information] = extracted_information
                        #
                        participant_a = {}
                        extracted_information[cdjf.participant_a] = participant_a
                        is_connected_to_model = self.add_protein_primary_information(participant_a, curr_obj.catalyst_str, curr_obj.catalyst_type,
                                                        curr_obj.catalyst_id, curr_obj.catalyst_state_str_list,
                                                        protein_name_idlist_map, proteins_in_model_lower_case, name_identifier_map_amrs) or is_connected_to_model
                        if cdj.is_identifier_prob:
                            identifier_prob_list.append(curr_obj.catalyst_id_prob)
                        #
                        participant_b = {}
                        extracted_information[cdjf.participant_b] = participant_b
                        participant_b[cdjf.entity_type] = dpt.complex
                        participant_b[cdjf.entities] = [{}, {}]
                        #
                        is_connected_to_model1 = self.add_protein_primary_information(participant_b[cdjf.entities][0], curr_obj.protein_1_str, curr_obj.protein_1_type,
                                                        curr_obj.protein_1_id, curr_obj.protein_1_state_str_list,
                                                        protein_name_idlist_map, proteins_in_model_lower_case, name_identifier_map_amrs)
                        is_connected_to_model = is_connected_to_model1 or is_connected_to_model
                        is_connected_to_model2 = self.add_protein_primary_information(participant_b[cdjf.entities][1], curr_obj.protein_2_str, curr_obj.protein_2_type,
                                                        curr_obj.protein_2_id, curr_obj.protein_2_state_str_list,
                                                        protein_name_idlist_map, proteins_in_model_lower_case, name_identifier_map_amrs)
                        is_connected_to_model = is_connected_to_model2 or is_connected_to_model
                        if is_connected_to_model1 or is_connected_to_model2:
                            participant_b[cdjf.in_model] = True
                        else:
                            participant_b[cdjf.in_model] = False
                        if cdj.is_identifier_prob:
                            identifier_prob_list.append(curr_obj.protein_1_id_prob)
                            identifier_prob_list.append(curr_obj.protein_2_id_prob)
                        #
                        if curr_obj.is_positive_catalyst:
                            extracted_information[cdjf.interaction_type] = dit.increases
                        else:
                            extracted_information[cdjf.interaction_type] = dit.decreases
                        extracted_information[cdjf.negative_information] = curr_obj.is_negative_information
                else:
                    print 'skipping complex disintegration interactions. current noise model can not model that. DARPA also does not need it. we can represent it as complex decrease instead if there is a catalyst'
                    continue
            elif isinstance(curr_obj, i.Interaction):
                #
                extracted_information = {}
                curr_json_map[cdjf.extracted_information] = extracted_information
                #
                if curr_obj.catalyst_str is not None and curr_obj.catalyst_str:
                    participant_a = {}
                    extracted_information[cdjf.participant_a] = participant_a
                    is_connected_to_model = self.add_protein_primary_information(participant_a, curr_obj.catalyst_str, curr_obj.catalyst_type,
                                                    curr_obj.catalyst_id, curr_obj.catalyst_state_str_list,
                                                    protein_name_idlist_map, proteins_in_model_lower_case, name_identifier_map_amrs) or is_connected_to_model
                    print 'participant_a', participant_a
                    if cdj.is_identifier_prob:
                        identifier_prob_list.append(curr_obj.catalyst_id_prob)
                else:
                    extracted_information[cdjf.participant_a] = None
                #
                participant_b = {}
                extracted_information[cdjf.participant_b] = participant_b
                is_connected_to_model = self.add_protein_primary_information(participant_b, curr_obj.protein_str, curr_obj.protein_type,
                                                curr_obj.protein_id, curr_obj.protein_state_str_list,
                                                protein_name_idlist_map, proteins_in_model_lower_case, name_identifier_map_amrs) or is_connected_to_model
                if cdj.is_identifier_prob:
                    identifier_prob_list.append(curr_obj.protein_id_prob)
                #
                mod_list = list(set(curr_obj.result_state_str_list) - set(curr_obj.protein_state_str_list))
                if not mod_list:
                    if not is_skip:
                        raise AssertionError
                    else:
                        continue
                if len(mod_list) == 1 and is_activity_type(mod_list[0]):
                    if curr_obj.is_positive_catalyst:
                        if not is_decrease_activity(mod_list[0]):
                            extracted_information[cdjf.interaction_type] = dit.increases_activity
                        else:
                            extracted_information[cdjf.interaction_type] = dit.decreases_activity
                    else:
                        if not is_decrease_activity(mod_list[0]):
                            extracted_information[cdjf.interaction_type] = dit.decreases_activity
                        else:
                            extracted_information[cdjf.interaction_type] = dit.increases_activity
                elif len(mod_list) == 1 and is_decrease_type(mod_list[0]):
                    if curr_obj.is_positive_catalyst:
                        extracted_information[cdjf.interaction_type] = dit.decreases
                    else:
                        extracted_information[cdjf.interaction_type] = dit.increases
                elif len(mod_list) == 1 and is_increase_type(mod_list[0]):
                    if curr_obj.is_positive_catalyst:
                        extracted_information[cdjf.interaction_type] = dit.increases
                    else:
                        extracted_information[cdjf.interaction_type] = dit.decreases
                elif self.all_mutation_type_state(mod_list):
                    continue
                elif self.all_loc_type_state(mod_list)[0]:
                    # ignoring catalyst type in translocation case
                    extracted_information[cdjf.interaction_type] = dit.translocates
                    mod_locs_only_list = remove_translocate_type_term_frm_list(mod_list)
                    if len(mod_locs_only_list) > 1:
                        continue
                    elif len(mod_locs_only_list) == 1:
                        extracted_information[cdjf.to_location_id] = get_GO_identifier_fr_loc_state(mod_locs_only_list[0])
                        extracted_information[cdjf.to_location_text] = mod_locs_only_list[0]
                    else:
                        extracted_information[cdjf.to_location_id] = ''
                        extracted_information[cdjf.to_location_text] = ''
                    prv_loc_list = self.all_loc_type_state(curr_obj.protein_state_str_list)[1]
                    if prv_loc_list:
                        non_loc_prv_state = list(set(curr_obj.protein_state_str_list) - set(prv_loc_list))
                        non_loc_features = self.get_features_list(non_loc_prv_state)
                        if non_loc_features is not None and non_loc_features:
                            participant_b[cdjf.features] = non_loc_features
                    prv_loc_only_list = remove_translocate_type_term_frm_list(prv_loc_list)
                    if len(prv_loc_only_list) > 1:
                        continue
                    elif len(prv_loc_only_list) == 1:
                        extracted_information[cdjf.from_location_id] = get_GO_identifier_fr_loc_state(prv_loc_only_list[0])
                        extracted_information[cdjf.from_location_text] = prv_loc_only_list[0]
                    else:
                        extracted_information[cdjf.from_location_id] = ''
                        extracted_information[cdjf.from_location_text] = ''
                else:
                    modifications = self.get_features_list(mod_list, is_feature_type=False)
                    print 'modifications', modifications
                    if modifications is not None and modifications:
                        if curr_obj.is_positive_catalyst:
                            extracted_information[cdjf.interaction_type] = dit.adds_modification
                        else:
                            extracted_information[cdjf.interaction_type] = dit.removes_modification
                        extracted_information[cdjf.modification] = modifications
                    else:
                        if not is_skip:
                            raise AssertionError
                        else:
                            continue
                extracted_information[cdjf.negative_information] = curr_obj.is_negative_information
                print 'curr_json_map', curr_json_map
            else:
                raise AssertionError
            if is_connected_to_model or (not cd.is_connected_to_model_required):
                json_interaction_objs.append(curr_json_map)
            else:
                if not self.is_model:
                    print 'not connected to model'
                continue
        if debug:
            print 'json_interaction_objs', json_interaction_objs
        if json_interaction_objs is not None and json_interaction_objs:
            # assign start time and end time to json objects
            if extraction_compute_time is not None:
                if amr_parsing_frm_txt_compute_time is None:
                    raise AssertionError
                # excluding file writing time which is negligible
                job_start_time = time.time()-amr_parsing_frm_txt_compute_time-extraction_compute_time
                job_end_time = time.time()
                num_json_objs = len(json_interaction_objs)
                compute_time_per_object = (job_end_time-job_start_time)/float(num_json_objs)
            else:
                job_start_time = time.time()
                compute_time_per_object = 1
            clock = job_start_time
            for curr_json_map in json_interaction_objs:
                curr_json_map[cdjf.reading_started] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(clock))
                clock += compute_time_per_object
                curr_json_map[cdjf.reading_complete] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(clock))
        return json_interaction_objs

    def update_relation_model(self, json_interaction_objs, model_json_objs):
        # update relation to model
        if model_json_objs is None or not model_json_objs:
            raise AssertionError
        for curr_json_map in json_interaction_objs:
            rm.compare_json_obj_to_model_json_objs(curr_json_map, model_json_objs)

    def write_json_index_cards(self, json_interaction_objs, directory_path=None, is_append=None):
        if (not ch.is_hpcc) or directory_path is not None:
            start_write_time = time.time()
            if directory_path is None:
                directory_path = '../biopax_model_interactions_json'
            if is_append is None:
                is_append = False
            if is_append:
                raise NotImplementedError
            else:
                if not os.path.exists(cap.absolute_path+directory_path):
                    # shutil.rmtree(cap.absolute_path+directory_path)
                    os.makedirs(cap.absolute_path+directory_path)
            pmc_cards_count_map = {}
            for curr_json_map in json_interaction_objs:
                curr_pmc_id = curr_json_map[cdjf.pmc_id]
                if curr_pmc_id is None or not curr_pmc_id:
                    curr_pmc_id = 'biopax_model'
                curr_directory_path = directory_path + '/' + curr_pmc_id
                if curr_pmc_id not in pmc_cards_count_map:
                    pmc_cards_count_map[curr_pmc_id] = 0
                    if os.path.exists(cap.absolute_path+curr_directory_path):
                        shutil.rmtree(cap.absolute_path+curr_directory_path)
                    os.makedirs(cap.absolute_path+curr_directory_path)
                pmc_cards_count_map[curr_pmc_id] += 1
                curr_index_card_file_path = curr_directory_path + '/' + 'index_card_' + str(pmc_cards_count_map[curr_pmc_id]) + '.json'
                curr_str = json.dumps(curr_json_map, ensure_ascii=True, sort_keys=True, indent=5)
                with open(cap.absolute_path+curr_index_card_file_path, 'w') as f:
                    f.write(curr_str)
            print 'Time to write json files was ', time.time()-start_write_time


class InteractionsSyntheticAMR:
    def __init__(self):
        self.count_joint = 0
        self.count_state = 0
        self.file_path_joint = './biopax_model_amrs/triplet_joint_subgraphs/'
        self.file_path_state = './biopax_model_amrs/protein_state_subgraphs/'

    def get_sentence(self, curr_interaction_obj):
        if curr_interaction_obj.text_sentence is not None and curr_interaction_obj.text_sentence.strip():
            sentence = curr_interaction_obj.text_sentence
        else:
            sentence = str(curr_interaction_obj)
        return sentence

    def add_protein_state(self, concept_str, protein_str, protein_state_str_list, data_state, sentence):
        assert concept_str is not None and concept_str
        assert protein_str is not None and protein_str
        for curr_state_term in protein_state_str_list:
            if curr_state_term is None or not curr_state_term.strip():
                continue
            self.count_state += 1
            curr_state_amr_info_map = self.gen_amr_protein_state(protein_str, concept_str, curr_state_term)
            path_key = self.file_path_state + 'protein_state_' + str(self.count_state)
            ead.nodes_to_dot(curr_state_amr_info_map['nodes'], path_key, sentence)
            #
            data_state[gefd.const_paths_map][path_key] = curr_state_amr_info_map['nodes']
            data_state[gefd.const_protein_state_tuples_map][path_key] = curr_state_amr_info_map['tuple']
            data_state[gefd.const_sentences_map][path_key] = sentence
            data_state[gefd.const_joint_labels_map][path_key] = 1

    def add_state_fr_all_proteins_of_interaction(self, curr_concept_term, curr_interaction_obj, data_state, sentence):
        if curr_interaction_obj.catalyst_str is not None and curr_interaction_obj.catalyst_str.strip():
            self.add_protein_state(curr_concept_term, curr_interaction_obj.catalyst_str, curr_interaction_obj.catalyst_state_str_list, data_state, sentence)
        if isinstance(curr_interaction_obj, i.Interaction):
            self.add_protein_state(curr_concept_term, curr_interaction_obj.protein_str, curr_interaction_obj.protein_state_str_list, data_state, sentence)
        elif isinstance(curr_interaction_obj, i.ComplexTypeInteraction):
            self.add_protein_state(curr_concept_term, curr_interaction_obj.protein_1_str, curr_interaction_obj.protein_1_state_str_list, data_state, sentence)
            self.add_protein_state(curr_concept_term, curr_interaction_obj.protein_2_str, curr_interaction_obj.protein_2_state_str_list, data_state, sentence)
        else:
            raise AssertionError

    def add_subgraph_from_state_change_interaction(self, curr_concept_term, curr_state_change_interaction_obj, data, data_state):
        curr_amr_info_map = self.gen_amr_triplet_joint(curr_concept_term, curr_state_change_interaction_obj.catalyst_str, curr_state_change_interaction_obj.protein_str)
        sentence = self.get_sentence(curr_state_change_interaction_obj)
        self.count_joint += 1
        path_key = self.file_path_joint+'state_change_'+str(self.count_joint)
        ead.nodes_to_dot(curr_amr_info_map['nodes'], path_key, sentence)
        #
        data[gefd.const_paths_map][path_key] = curr_amr_info_map['nodes']
        data[gefd.const_interaction_tuples_map][path_key] = curr_amr_info_map['tuple']
        data[gefd.const_sentences_map][path_key] = sentence
        data[gefd.const_joint_labels_map][path_key] = 1
        #
        self.add_state_fr_all_proteins_of_interaction(curr_concept_term, curr_state_change_interaction_obj, data_state, sentence)

    def add_subgraph_from_complex_form_type_interaction(self, curr_complex_formation_obj, data, data_state):
        if curr_complex_formation_obj.complex_interaction_str is not None and curr_complex_formation_obj.complex_interaction_str.strip():
            interaction_type = curr_complex_formation_obj.complex_interaction_str.strip()
        else:
            interaction_type = 'bind'
        curr_amr_info_map = self.gen_amr_triplet_joint(interaction_type, curr_complex_formation_obj.catalyst_str, curr_complex_formation_obj.protein_1_str, curr_complex_formation_obj.protein_2_str)
        sentence = self.get_sentence(curr_complex_formation_obj)
        self.count_joint += 1
        path_key = self.file_path_joint+'complex_form_type_'+str(self.count_joint)
        ead.nodes_to_dot(curr_amr_info_map['nodes'], path_key, sentence)
        #
        data[gefd.const_paths_map][path_key] = curr_amr_info_map['nodes']
        data[gefd.const_interaction_tuples_map][path_key] = curr_amr_info_map['tuple']
        data[gefd.const_sentences_map][path_key] = sentence
        data[gefd.const_joint_labels_map][path_key] = 1
        #
        self.add_state_fr_all_proteins_of_interaction(interaction_type, curr_complex_formation_obj, data_state, sentence)

    def gen_amr_triplet_joint(self, concept_str, catalyst_str, protein_str, protein2_str=None):
        curr_nodes_list = []
        #
        assert concept_str is not None and concept_str
        concept_node = ead.Node(id='c1', name=concept_str)
        concept_node.color = 'blue'
        concept_str = None
        curr_nodes_list.append(concept_node)
        #
        if catalyst_str is not None and catalyst_str:
            catalyst_node = ead.Node(id='c2', name=catalyst_str)
            catalyst_node.type = 'enzyme'
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
        concept_node.add_parent_child_relationship(protein_node, cjs.hasProtein)
        if catalyst_node is not None:
            concept_node.add_parent_child_relationship(catalyst_node, cjs.hasCatalyst)
        if protein2_node is not None:
            concept_node.add_parent_child_relationship(protein2_node, cjs.hasProtein2)
        #
        curr_amr_map['tuple'] = tuple(curr_triplet_tuple)
        curr_amr_map['nodes'] = curr_nodes_list
        return curr_amr_map

    def gen_amr_protein_state(self, protein_str, concept_str, state_str):
        assert protein_str is not None and protein_str.strip()
        assert concept_str is not None and concept_str.strip()
        assert state_str is not None and state_str.strip()
        #
        curr_nodes_list = []
        #
        protein_node = ead.Node(id='p1', name=protein_str)
        protein_node.type = 'protein'
        protein_node.color = 'green'
        protein_str = None
        curr_nodes_list.append(protein_node)
        #
        concept_node = ead.Node(id='c1', name=concept_str)
        concept_node.color = 'red'
        concept_str = None
        curr_nodes_list.append(concept_node)
        #
        state_node = ead.Node(id='s1', name=state_str)
        state_node.color = '#976850'
        state_str = None
        curr_nodes_list.append(state_node)
        #
        curr_amr_map = {}
        #
        curr_protein_concept_state_tuple = [protein_node, concept_node, state_node]
        #
        protein_node.add_parent_child_relationship(concept_node, cpss.relatedToConcept)
        protein_node.add_parent_child_relationship(state_node, cpss.hasState)
        #
        curr_amr_map['tuple'] = tuple(curr_protein_concept_state_tuple)
        curr_amr_map['nodes'] = curr_nodes_list
        return curr_amr_map

    def gen_amr_subgraphs_frm_model_interactions(self, interactions_map):
        data = {}
        data[gefd.const_paths_map] = {}
        data[gefd.const_interaction_tuples_map] = {}
        data[gefd.const_sentences_map] = {}
        data[gefd.const_joint_labels_map] = {}
        #
        data_state = {}
        data_state[gefd.const_paths_map] = {}
        data_state[gefd.const_protein_state_tuples_map] = {}
        data_state[gefd.const_sentences_map] = {}
        data_state[gefd.const_joint_labels_map] = {}
        for curr_interaction_obj in interactions_map['state_change']:
            mod_list = list(set(curr_interaction_obj.result_state_str_list) - set(curr_interaction_obj.protein_state_str_list))
            darpa_mod_features_map, other_features_list = get_features_map(mod_list, is_feature_type=True)
            curr_concepts_list = darpa_mod_features_map.keys()+other_features_list
            for curr_concept_term in curr_concepts_list:
                self.add_subgraph_from_state_change_interaction(curr_concept_term, curr_interaction_obj, data, data_state)
                # add proteins state synthetic amrs here
                if is_loc_type(curr_concept_term):
                    self.add_subgraph_from_state_change_interaction('translocate', curr_interaction_obj, data, data_state)
        for curr_complex_formation in interactions_map['complex']:
            self.add_subgraph_from_complex_form_type_interaction(curr_complex_formation, data, data_state)
        gefd.dump_pickle_data_joint_model(data, is_extend=False)
        gefd.dump_pickle_data_protein_state_model(data_state, is_extend=False)

