import csv
import copy
import re
import urllib
import interactions as i
import read_against_model as ram
import constants_absolute_path as cap
from config import *
import interactions_to_json_or_amrs as ija
import time
import uniprot_mapping_obj as umo
import protein_identifiers_darpa_obj as pido
import json
from config_console_output import *
import config_hpcc as ch
import identifiers_owl_obj as ioo
import identifiers_map_owl_obj as imoo


is_skip = True
is_add_interaction_type_to_result_state = False

model_folder_path = './csv_biopax_model/'

# controller
const_csv_header_control = 'control'
const_csv_header_control_control_type = 'control_control_type'
const_csv_header_control_controller = 'control_controller'
const_csv_header_control_controller_name = 'control_controller_name'
const_csv_header_control_controller_er = 'control_controller_er'
const_csv_header_control_controller_cellloc = 'control_controller_cellloc'
const_csv_header_control_controller_cellloc_term = 'control_controller_cellloc_term'
const_csv_header_control_controller_feature = 'control_controller_feature'
const_csv_header_control_controller_feature_modf_type = 'control_controller_feature_modf_type'
const_csv_header_control_controller_feature_modf_type_term = 'control_controller_feature_modf_type_term'
const_csv_header_control_controller_feature_loc = 'control_controller_feature_loc'
const_csv_header_control_controller_feature_loc_seqpos = 'control_controller_feature_loc_seqpos'
const_csv_header_control_controller_component = 'control_controller_component'
const_csv_header_control_controller_component_name = 'control_controller_component_name'
const_csv_header_control_controller_component_er = 'control_controller_component_er'
const_csv_header_control_controller_component_cellloc = 'control_controller_component_cellloc'
const_csv_header_control_controller_component_cellloc_term = 'control_controller_component_cellloc_term'
const_csv_header_control_controller_component_feature = 'control_controller_component_feature'
const_csv_header_control_controller_component_feature_modf_type = 'control_controller_component_feature_modf_type'
const_csv_header_control_controller_component_feature_modf_type_term = 'control_controller_component_feature_modf_type_term'
const_csv_header_control_controller_component_feature_loc = 'control_controller_component_feature_loc'
const_csv_header_control_controller_component_feature_loc_seqpos = 'control_controller_component_feature_loc_seqpos'
const_csv_header_control_controlled = 'control_controlled'


#
const_is_positive_catalyst = 'is_positive_catalyst'
const_catalyst_str = 'catalyst_str'
const_catalyst_type = 'catalyst_type'
const_catalyst_id = 'catalyst_id'
const_catalyst_state_str_list = 'catalyst_state_str_list'
const_controlled = 'controlled'

# controlled
const_csv_header_controlled = 'controlled'
const_csv_header_controlled_type = 'controlled_type'
const_csv_header_controlled_name = 'controlled_name'
const_csv_header_controlled_interaction_type = 'controlled_interaction_type'
const_csv_header_controlled_interaction_type_term = 'controlled_interaction_type_term'
const_csv_header_controlled_is_spontaneous = 'controlled_is_spontaneous'
const_csv_header_controlled_left = 'controlled_left'
const_csv_header_controlled_left_name = 'controlled_left_name'
const_csv_header_controlled_left_er = 'controlled_left_er'
const_csv_header_controlled_left_cellloc = 'controlled_left_cellloc'
const_csv_header_controlled_left_cellloc_term = 'controlled_left_cellloc_term'
const_csv_header_controlled_left_feature = 'controlled_left_feature'
const_csv_header_controlled_left_feature_modf_type = 'controlled_left_feature_modf_type'
const_csv_header_controlled_left_feature_modf_type_term = 'controlled_left_feature_modf_type_term'
const_csv_header_controlled_left_feature_loc = 'controlled_left_feature_loc'
const_csv_header_controlled_left_feature_loc_seqpos = 'controlled_left_feature_loc_seqpos'
const_csv_header_controlled_left_component = 'controlled_left_component'
const_csv_header_controlled_left_component_name = 'controlled_left_component_name'
const_csv_header_controlled_left_component_er = 'controlled_left_component_er'
const_csv_header_controlled_left_component_cellloc = 'controlled_left_component_cellloc'
const_csv_header_controlled_left_component_cellloc_term = 'controlled_left_component_cellloc_term'
const_csv_header_controlled_left_component_feature = 'controlled_left_component_feature'
const_csv_header_controlled_left_component_feature_modf_type = 'controlled_left_component_feature_modf_type'
const_csv_header_controlled_left_component_feature_modf_type_term = 'controlled_left_component_feature_modf_type_term'
const_csv_header_controlled_left_component_feature_loc = 'controlled_left_component_feature_loc'
const_csv_header_controlled_left_component_feature_loc_seqpos = 'controlled_left_component_feature_loc_seqpos'
const_csv_header_controlled_right = 'controlled_right'
const_csv_header_controlled_right_name = 'controlled_right_name'
const_csv_header_controlled_right_er = 'controlled_right_er'
const_csv_header_controlled_right_cellloc = 'controlled_right_cellloc'
const_csv_header_controlled_right_cellloc_term = 'controlled_right_cellloc_term'
const_csv_header_controlled_right_feature = 'controlled_right_feature'
const_csv_header_controlled_right_feature_modf_type = 'controlled_right_feature_modf_type'
const_csv_header_controlled_right_feature_modf_type_term = 'controlled_right_feature_modf_type_term'
const_csv_header_controlled_right_feature_loc = 'controlled_right_feature_loc'
const_csv_header_controlled_right_feature_loc_seqpos = 'controlled_right_feature_loc_seqpos'
const_csv_header_controlled_right_component = 'controlled_right_component'
const_csv_header_controlled_right_component_name = 'controlled_right_component_name'
const_csv_header_controlled_right_component_er = 'controlled_right_component_er'
const_csv_header_controlled_right_component_cellloc = 'controlled_right_component_cellloc'
const_csv_header_controlled_right_component_cellloc_term = 'controlled_right_component_cellloc_term'
const_csv_header_controlled_right_component_feature = 'controlled_right_component_feature'
const_csv_header_controlled_right_component_feature_modf_type = 'controlled_right_component_feature_modf_type'
const_csv_header_controlled_right_component_feature_modf_type_term = 'controlled_right_component_feature_modf_type_term'
const_csv_header_controlled_right_component_feature_loc = 'controlled_right_component_feature_loc'
const_csv_header_controlled_right_component_feature_loc_seqpos = 'controlled_right_component_feature_loc_seqpos'


const_biopax_type = 'biopax_type'
#
const_sentence = 'sentence'
const_interaction_type = 'interaction_type'
#
const_protein1_left = 'protein1_left'
const_protein2_left = 'protein2_left'
#
const_protein1_right = 'protein1_right'
const_protein2_right = 'protein2_right'
#
const_protein1_left_str = 'protein1_left_str'
const_protein2_left_str = 'protein2_left_str'
#
const_protein1_right_str = 'protein1_right_str'
const_protein2_right_str = 'protein2_right_str'
#
const_protein1_left_type = 'protein1_left_type'
const_protein2_left_type = 'protein2_left_type'
#
const_protein1_right_type = 'protein1_right_type'
const_protein2_right_type = 'protein2_right_type'
#
const_protein1_left_id = 'protein1_left_id'
const_protein2_left_id = 'protein2_left_id'
#
const_protein1_right_id = 'protein1_right_id'
const_protein2_right_id = 'protein2_right_id'
#
const_protein1_left_state_str_list = 'protein1_left_state_str_list'
const_protein2_left_state_str_list = 'protein2_left_state_str_list'
const_protein1_right_state_str_list = 'protein1_right_state_str_list'
const_protein2_right_state_str_list = 'protein2_right_state_str_list'


def extract_type_frm_biopax_url(url_str):
    m = re.search('#(.+?)_', url_str)
    if m is not None:
        return m.group(1)


def extract_id_frm_identifier_url(url_str):
    print 'url_str', url_str
    if url_str is None or not url_str:
        return None
    url_str_list = url_str.split(',')
    ids_list = []
    for curr_url_str in url_str_list:
        curr_url_str += ' '
        print 'curr_url_str', curr_url_str
        if curr_url_str is not None and curr_url_str:
            curr_url_str = urllib.unquote(curr_url_str).decode('utf8')
            if 'purl' in curr_url_str:
                continue
            #
            m = re.findall('chebi(.+?) ', curr_url_str)
            if m is not None and m:
                if len(m) != 1:
                    raise AssertionError
                ids_list.append(m[0].replace('/', '').strip())
            else:
                m = re.findall('uniprot(.+?) ', curr_url_str)
                if m is not None and m:
                    if len(m) != 1:
                        raise AssertionError
                    ids_list.append('Uniprot:' + m[0].replace('/', '').strip())
                else:
                    pass
                    # raise AssertionError
    if ids_list:
        for i in range(len(ids_list)):
            curr_id = ids_list[i]
            curr_id_mapping = umo.um_obj.get_mapping(curr_id)
            if curr_id_mapping is not None and curr_id_mapping:
                ids_list[i] = curr_id_mapping
    if ids_list:
        print 'ids_list', ids_list
        return ','.join(ids_list)


def parse_controller(controller_path=None):
    if controller_path is None:
        controller_path = model_folder_path+'controller'+'.csv'
    count = 0
    control_interactions_raw = {}
    controller_control_map = {}
    with open(cap.absolute_path+controller_path, 'rU') as cf:
        reader = csv.DictReader(cf)
        for curr_row in reader:
            count += 1
            print curr_row
            #parse the row into a raw map of interactions
            curr_control_key = curr_row[const_csv_header_control]
            if curr_control_key not in control_interactions_raw:
                control_interactions_raw[curr_control_key] = {}
                #
                #  is_catalyst
                is_catalyst = None
                if 'INHIBITION' in curr_row[const_csv_header_control_control_type].strip().upper():
                    is_catalyst = False
                elif 'ACTIVATION' in curr_row[const_csv_header_control_control_type].strip().upper():
                    is_catalyst = True
                elif not curr_row[const_csv_header_control_control_type].strip():
                    is_catalyst = True #default case is True
                else:
                    print 'curr_row[const_csv_header_control_control_type].strip()', curr_row[const_csv_header_control_control_type].strip()
                    raise NotImplementedError
                control_interactions_raw[curr_control_key][const_is_positive_catalyst] = is_catalyst
                #
                # catalyst_str
                if curr_row[const_csv_header_control_controller_name].strip():
                    control_interactions_raw[curr_control_key][const_catalyst_str] = curr_row[const_csv_header_control_controller_name].strip()
                    control_interactions_raw[curr_control_key][const_catalyst_type] = extract_type_frm_biopax_url(curr_row[const_csv_header_control_controller].strip())
                else:
                    raise AssertionError
                #
                # catalyst_id
                if curr_row[const_csv_header_control_controller_er].strip():
                    control_interactions_raw[curr_control_key][const_catalyst_id] = curr_row[const_csv_header_control_controller_er].strip()
                elif curr_row[const_csv_header_control_controller_component_er].strip():
                    control_interactions_raw[curr_control_key][const_catalyst_id] = curr_row[const_csv_header_control_controller_component_er].strip()
                else:
                    control_interactions_raw[curr_control_key][const_catalyst_id] = ''
                #
                #catalyst_state_str_list
                catalyst_state_str_list = []
                if curr_row[const_csv_header_control_controller_cellloc_term].strip():
                    catalyst_state_str_list.append(curr_row[const_csv_header_control_controller_cellloc_term].strip())
                if curr_row[const_csv_header_control_controller_feature].strip():
                    if curr_row[const_csv_header_control_controller_feature_modf_type_term].strip():
                        catalyst_state_str_list.append(curr_row[const_csv_header_control_controller_feature_modf_type_term].replace('residue modification,', '').strip())
                    if curr_row[const_csv_header_control_controller_feature_loc_seqpos].strip():
                        catalyst_state_str_list.append(curr_row[const_csv_header_control_controller_feature_loc_seqpos].strip())
                # component level state and complex state is merged
                if curr_row[const_csv_header_control_controller_component].strip():
                    if curr_row[const_csv_header_control_controller_component_cellloc_term].strip():
                        catalyst_state_str_list.append(curr_row[const_csv_header_control_controller_component_cellloc_term].strip())
                    if curr_row[const_csv_header_control_controller_component_feature].strip():
                        if curr_row[const_csv_header_control_controller_component_feature_modf_type_term].strip():
                            catalyst_state_str_list.append(curr_row[const_csv_header_control_controller_component_feature_modf_type_term].replace('residue modification,', '').strip())
                        if curr_row[const_csv_header_control_controller_component_feature_loc_seqpos].strip():
                            catalyst_state_str_list.append(curr_row[const_csv_header_control_controller_component_feature_loc_seqpos].strip())
                control_interactions_raw[curr_control_key][const_catalyst_state_str_list] = list(set(catalyst_state_str_list))
                #controlled
                control_interactions_raw[curr_control_key][const_controlled] = curr_row[const_csv_header_control_controlled]
                if curr_row[const_csv_header_control_controlled] not in controller_control_map:
                    controller_control_map[curr_row[const_csv_header_control_controlled]] = []
                controller_control_map[curr_row[const_csv_header_control_controlled]].append(curr_row[const_csv_header_control])
                assert len(controller_control_map[curr_row[const_csv_header_control_controlled]]) == len(set(controller_control_map[curr_row[const_csv_header_control_controlled]]))
            else:
                if not curr_row[const_csv_header_control_controller_er].strip() and curr_row[const_csv_header_control_controller_component_er].strip():
                    if curr_row[const_csv_header_control_controller_component_er].strip() not in control_interactions_raw[curr_control_key][const_catalyst_id]:
                        control_interactions_raw[curr_control_key][const_catalyst_id] += ', ' + curr_row[const_csv_header_control_controller_component_er].strip()
                #
                #catalyst_state_str_list
                catalyst_state_str_list = []
                # component level state and complex state is merged
                if curr_row[const_csv_header_control_controller_component].strip():
                    if curr_row[const_csv_header_control_controller_component_cellloc_term].strip():
                        catalyst_state_str_list.append(curr_row[const_csv_header_control_controller_component_cellloc_term].strip())
                    if curr_row[const_csv_header_control_controller_component_feature].strip():
                        if curr_row[const_csv_header_control_controller_component_feature_modf_type_term].strip():
                            catalyst_state_str_list.append(curr_row[const_csv_header_control_controller_component_feature_modf_type_term].replace('residue modification,', '').strip())
                        if curr_row[const_csv_header_control_controller_component_feature_loc_seqpos].strip():
                            catalyst_state_str_list.append(curr_row[const_csv_header_control_controller_component_feature_loc_seqpos].strip())
                control_interactions_raw[curr_control_key][const_catalyst_state_str_list] += list(set(catalyst_state_str_list))
                control_interactions_raw[curr_control_key][const_catalyst_state_str_list] = list(set(control_interactions_raw[curr_control_key][const_catalyst_state_str_list]))
        print 'Total count of read rows was {}'.format(count)
        print control_interactions_raw
        print controller_control_map
        return control_interactions_raw, controller_control_map


def parse_controlled(controlled_path=None):
    def get_left_state_str_list(curr_row, component_only=False):
        state_str_list = []
        if curr_row[const_csv_header_controlled_left_cellloc_term].strip():
            state_str_list.append(curr_row[const_csv_header_controlled_left_cellloc_term].strip())
        if curr_row[const_csv_header_controlled_left_feature].strip():
            if curr_row[const_csv_header_controlled_left_feature_modf_type_term].strip():
                state_str_list.append(curr_row[const_csv_header_controlled_left_feature_modf_type_term].replace('residue modification,', '').strip())
            if curr_row[const_csv_header_controlled_left_feature_loc_seqpos].strip():
                state_str_list.append(curr_row[const_csv_header_controlled_left_feature_loc_seqpos].strip())
        if curr_row[const_csv_header_controlled_left_component].strip():
            if curr_row[const_csv_header_controlled_left_component_cellloc_term].strip():
                state_str_list.append(curr_row[const_csv_header_controlled_left_component_cellloc_term].strip())
            if curr_row[const_csv_header_controlled_left_component_feature].strip():
                if curr_row[const_csv_header_controlled_left_component_feature_modf_type_term].strip():
                    state_str_list.append(curr_row[const_csv_header_controlled_left_component_feature_modf_type_term].replace('residue modification,', '').strip())
                if curr_row[const_csv_header_controlled_left_component_feature_loc_seqpos].strip():
                    state_str_list.append(curr_row[const_csv_header_controlled_left_component_feature_loc_seqpos].strip())
        return list(set(state_str_list))

    def get_right_state_str_list(curr_row):
        state_str_list = []
        if curr_row[const_csv_header_controlled_right_cellloc_term].strip():
            state_str_list.append(curr_row[const_csv_header_controlled_right_cellloc_term].strip())
        if curr_row[const_csv_header_controlled_right_feature].strip():
            if curr_row[const_csv_header_controlled_right_feature_modf_type_term].strip():
                state_str_list.append(curr_row[const_csv_header_controlled_right_feature_modf_type_term].replace('residue modification,', '').strip())
            if curr_row[const_csv_header_controlled_right_feature_loc_seqpos].strip():
                state_str_list.append(curr_row[const_csv_header_controlled_right_feature_loc_seqpos].strip())
        if curr_row[const_csv_header_controlled_right_component].strip():
            if curr_row[const_csv_header_controlled_right_component_cellloc_term].strip():
                state_str_list.append(curr_row[const_csv_header_controlled_right_component_cellloc_term].strip())
            if curr_row[const_csv_header_controlled_right_component_feature].strip():
                if curr_row[const_csv_header_controlled_right_component_feature_modf_type_term].strip():
                    state_str_list.append(curr_row[const_csv_header_controlled_right_component_feature_modf_type_term].replace('residue modification,', '').strip())
                if curr_row[const_csv_header_controlled_right_component_feature_loc_seqpos].strip():
                    state_str_list.append(curr_row[const_csv_header_controlled_right_component_feature_loc_seqpos].strip())
        return list(set(state_str_list))

    def get_left_id(curr_row, component_only=False):
        if curr_row[const_csv_header_controlled_left_er].strip() and (not component_only):
            id = curr_row[const_csv_header_controlled_left_er].strip()
        elif curr_row[const_csv_header_controlled_left_component_er].strip():
            id = curr_row[const_csv_header_controlled_left_component_er].strip()
        else:
            id = ''
        return id

    def get_right_id(curr_row, component_only=False):
        if curr_row[const_csv_header_controlled_right_er].strip() and (not component_only):
            id = curr_row[const_csv_header_controlled_right_er].strip()
        elif curr_row[const_csv_header_controlled_right_component_er].strip():
            id = curr_row[const_csv_header_controlled_right_component_er].strip()
        else:
            id = ''
        return id

    if controlled_path is None:
        controlled_path = model_folder_path+'controlled'+'.csv'
    count = 0
    controlled_interactions_raw = {}
    with open(cap.absolute_path+controlled_path, 'rU') as cf:
        reader = csv.DictReader(cf)
        for curr_row in reader:
            count += 1
            print curr_row
            curr_controlled_key = curr_row[const_controlled]
            print 'curr_controlled_key', curr_controlled_key
            if curr_controlled_key not in controlled_interactions_raw:
                controlled_interactions_raw[curr_controlled_key] = {}
                if debug:
                    print 'controlled_interactions_raw[curr_controlled_key]', controlled_interactions_raw[curr_controlled_key]
                #
                # biopax type
                controlled_interactions_raw[curr_controlled_key][const_biopax_type] = curr_row[const_csv_header_controlled_type].strip()
                #
                # sentence
                controlled_interactions_raw[curr_controlled_key][const_sentence] = curr_row[const_csv_header_controlled_name].strip()
                # interaction type
                controlled_interactions_raw[curr_controlled_key][const_interaction_type] = curr_row[const_csv_header_controlled_interaction_type_term].strip()
                # protein 1 left
                assert curr_row[const_csv_header_controlled_left].strip()
                controlled_interactions_raw[curr_controlled_key][const_protein1_left] = curr_row[const_csv_header_controlled_left].strip()
                # protein1 left name
                assert curr_row[const_csv_header_controlled_left_name].strip()
                controlled_interactions_raw[curr_controlled_key][const_protein1_left_str] = curr_row[const_csv_header_controlled_left_name].strip()
                controlled_interactions_raw[curr_controlled_key][const_protein1_left_type] = extract_type_frm_biopax_url(curr_row[const_csv_header_controlled_left].strip())
                # protein1 left id
                controlled_interactions_raw[curr_controlled_key][const_protein1_left_id] = get_left_id(curr_row)
                # protein1 left state
                controlled_interactions_raw[curr_controlled_key][const_protein1_left_state_str_list] = get_left_state_str_list(curr_row)
                if debug:
                    print 'controlled_interactions_raw[curr_controlled_key]', controlled_interactions_raw[curr_controlled_key]
                # protein 1 right
                if curr_row[const_csv_header_controlled_right].strip():
                    controlled_interactions_raw[curr_controlled_key][const_protein1_right] = curr_row[const_csv_header_controlled_right].strip()
                    # protein1 right name
                    assert curr_row[const_csv_header_controlled_right_name].strip()
                    controlled_interactions_raw[curr_controlled_key][const_protein1_right_str] = curr_row[const_csv_header_controlled_right_name].strip()
                    controlled_interactions_raw[curr_controlled_key][const_protein1_right_type] = extract_type_frm_biopax_url(curr_row[const_csv_header_controlled_right])
                    # protein1 right id
                    controlled_interactions_raw[curr_controlled_key][const_protein1_right_id] = get_right_id(curr_row)
                    # protein1 right state
                    controlled_interactions_raw[curr_controlled_key][const_protein1_right_state_str_list] = get_right_state_str_list(curr_row)
                    if debug:
                        print 'controlled_interactions_raw[curr_controlled_key]', controlled_interactions_raw[curr_controlled_key]
                else:
                    assert not curr_row[const_csv_header_controlled_right].strip()
                    assert not curr_row[const_csv_header_controlled_right_name].strip()
                    assert not curr_row[const_csv_header_controlled_right_er].strip()
                    assert not curr_row[const_csv_header_controlled_right_component_er].strip()
                    assert not curr_row[const_csv_header_controlled_right_cellloc].strip()
                    assert not curr_row[const_csv_header_controlled_right_component_cellloc].strip()
                    assert not curr_row[const_csv_header_controlled_right_feature].strip()
                    assert not curr_row[const_csv_header_controlled_right_component_feature].strip()
            else:
                assert curr_row[const_csv_header_controlled_left].strip()
                if curr_row[const_csv_header_controlled_left].strip() != controlled_interactions_raw[curr_controlled_key][const_protein1_left]:
                    # it means it is protein2 on left side
                    if const_protein2_left not in controlled_interactions_raw[curr_controlled_key]:
                        controlled_interactions_raw[curr_controlled_key][const_protein2_left] = curr_row[const_csv_header_controlled_left].strip()
                        assert curr_row[const_csv_header_controlled_left_name].strip()
                        controlled_interactions_raw[curr_controlled_key][const_protein2_left_str] = curr_row[const_csv_header_controlled_left_name].strip()
                        controlled_interactions_raw[curr_controlled_key][const_protein2_left_type] = extract_type_frm_biopax_url(curr_row[const_csv_header_controlled_left].strip())
                        controlled_interactions_raw[curr_controlled_key][const_protein2_left_id] = get_left_id(curr_row)
                        controlled_interactions_raw[curr_controlled_key][const_protein2_left_state_str_list] = get_left_state_str_list(curr_row)
                        if debug:
                            print 'controlled_interactions_raw[curr_controlled_key]', controlled_interactions_raw[curr_controlled_key]
                    else:
                        if not curr_row[const_csv_header_controlled_left_er]:
                            controlled_interactions_raw[curr_controlled_key][const_protein2_left_id] = ',' + get_left_id(curr_row, component_only=True)
                        controlled_interactions_raw[curr_controlled_key][const_protein2_left_state_str_list] += get_left_state_str_list(curr_row)
                        controlled_interactions_raw[curr_controlled_key][const_protein2_left_state_str_list] = list(set(controlled_interactions_raw[curr_controlled_key][const_protein2_left_state_str_list]))
                        if debug:
                            print 'controlled_interactions_raw[curr_controlled_key]', controlled_interactions_raw[curr_controlled_key]
                else:
                    if not curr_row[const_csv_header_controlled_left_er]:
                        controlled_interactions_raw[curr_controlled_key][const_protein1_left_id] = ',' + get_left_id(curr_row, component_only=True)
                    controlled_interactions_raw[curr_controlled_key][const_protein1_left_state_str_list] += get_left_state_str_list(curr_row)
                    controlled_interactions_raw[curr_controlled_key][const_protein1_left_state_str_list] = list(set(controlled_interactions_raw[curr_controlled_key][const_protein1_left_state_str_list]))
                    if debug:
                        print 'controlled_interactions_raw[curr_controlled_key]', controlled_interactions_raw[curr_controlled_key]
                if curr_row[const_csv_header_controlled_right].strip():
                    if curr_row[const_csv_header_controlled_right].strip() != controlled_interactions_raw[curr_controlled_key][const_protein1_right]:
                        #it means it is protein 2 right hand side
                        if const_protein2_right not in controlled_interactions_raw[curr_controlled_key]:
                            controlled_interactions_raw[curr_controlled_key][const_protein2_right] = curr_row[const_csv_header_controlled_right].strip()
                            assert curr_row[const_csv_header_controlled_right_name].strip()
                            controlled_interactions_raw[curr_controlled_key][const_protein2_right_str] = curr_row[const_csv_header_controlled_right_name].strip()
                            controlled_interactions_raw[curr_controlled_key][const_protein2_right_type] = extract_type_frm_biopax_url(curr_row[const_csv_header_controlled_right].strip())
                            controlled_interactions_raw[curr_controlled_key][const_protein2_right_id] = get_right_id(curr_row)
                            controlled_interactions_raw[curr_controlled_key][const_protein2_right_state_str_list] = get_right_state_str_list(curr_row)
                            if debug:
                                print 'controlled_interactions_raw[curr_controlled_key]', controlled_interactions_raw[curr_controlled_key]
                        else:
                            if not curr_row[const_csv_header_controlled_right_er]:
                                controlled_interactions_raw[curr_controlled_key][const_protein2_right_id] = ',' + get_right_id(curr_row, component_only=True)
                            controlled_interactions_raw[curr_controlled_key][const_protein2_right_state_str_list] += get_right_state_str_list(curr_row)
                            controlled_interactions_raw[curr_controlled_key][const_protein2_right_state_str_list] = list(set(controlled_interactions_raw[curr_controlled_key][const_protein2_right_state_str_list]))
                            if debug:
                                print 'controlled_interactions_raw[curr_controlled_key]', controlled_interactions_raw[curr_controlled_key]
                    else:
                        if not curr_row[const_csv_header_controlled_right_er]:
                            controlled_interactions_raw[curr_controlled_key][const_protein1_right_id] = ',' + get_right_id(curr_row, component_only=True)
                        controlled_interactions_raw[curr_controlled_key][const_protein1_right_state_str_list] += get_right_state_str_list(curr_row)
                        controlled_interactions_raw[curr_controlled_key][const_protein1_right_state_str_list] = list(set(controlled_interactions_raw[curr_controlled_key][const_protein1_right_state_str_list]))
                        if debug:
                            print 'controlled_interactions_raw[curr_controlled_key]', controlled_interactions_raw[curr_controlled_key]
                else:
                    assert not curr_row[const_csv_header_controlled_right].strip()
                    assert not curr_row[const_csv_header_controlled_right_name].strip()
                    assert not curr_row[const_csv_header_controlled_right_er].strip()
                    assert not curr_row[const_csv_header_controlled_right_component_er].strip()
                    assert not curr_row[const_csv_header_controlled_right_cellloc].strip()
                    assert not curr_row[const_csv_header_controlled_right_component_cellloc].strip()
                    assert not curr_row[const_csv_header_controlled_right_feature].strip()
                    assert not curr_row[const_csv_header_controlled_right_component_feature].strip()
    print controlled_interactions_raw
    return controlled_interactions_raw


def get_interactions_objs_from_raw(control_interactions_raw, controller_control_map, controlled_interactions_raw):
    def add_control_info_to_interaction_obj(curr_obj, curr_controller):
        curr_obj.catalyst_str = control_interactions_raw[curr_controller][const_catalyst_str]
        curr_obj.catalyst_state_str_list = control_interactions_raw[curr_controller][const_catalyst_state_str_list]
        curr_obj.is_positive_catalyst = control_interactions_raw[curr_controller][const_is_positive_catalyst]
        curr_obj.catalyst_id = control_interactions_raw[curr_controller][const_catalyst_id]

    print 'No. of controlled in controller_control_map is ', len(controller_control_map)
    controlled_list = controller_control_map.keys()
    print 'No. of controlled in controlled_interactions_raw is ', len(controlled_interactions_raw)
    controlled_list += controlled_interactions_raw.keys()
    controlled_list = list(set(controlled_list))
    print 'No. of unique in merged controlled_list is ', len(controlled_list)
    interactions_list = []
    for curr_controlled in controlled_list:
        if curr_controlled not in controlled_interactions_raw:
            continue
        print 'curr_controlled', curr_controlled
        print 'controlled_interactions_raw[curr_controlled]', controlled_interactions_raw[curr_controlled]
        #first processing the controlled part and then controller part (there maybe multiple controllers for a single controlled)
        is_complex_type = False
        if (const_protein1_left in controlled_interactions_raw[curr_controlled]
            and controlled_interactions_raw[curr_controlled][const_protein1_left])\
                and (const_protein2_left in controlled_interactions_raw[curr_controlled]
                     and controlled_interactions_raw[curr_controlled][const_protein2_left]):
            is_complex_type = True
            is_left_to_right = True
            if (const_protein2_right in controlled_interactions_raw[curr_controlled] and controlled_interactions_raw[curr_controlled][const_protein2_right]):
                if not is_skip:
                    raise AssertionError
                else:
                    continue
        elif (const_protein1_right in controlled_interactions_raw[curr_controlled]
              and controlled_interactions_raw[curr_controlled][const_protein1_right]) \
                and (const_protein2_right in controlled_interactions_raw[curr_controlled]
                     and controlled_interactions_raw[curr_controlled][const_protein2_right]):
            is_complex_type = True
            is_left_to_right = False
            if (const_protein2_left in controlled_interactions_raw[curr_controlled] and controlled_interactions_raw[curr_controlled][const_protein2_left]):
                if not is_skip:
                    raise AssertionError
                else:
                    continue
        if is_complex_type:
            if is_left_to_right:
                complex_type_interaction_obj =\
                    i.ComplexTypeInteraction(
                        protein_1_str=controlled_interactions_raw[curr_controlled][const_protein1_left_str],
                        protein_1_state_str_list=controlled_interactions_raw[curr_controlled][const_protein1_left_state_str_list],
                        protein_2_str=controlled_interactions_raw[curr_controlled][const_protein2_left_str],
                        protein_2_state_str_list=controlled_interactions_raw[curr_controlled][const_protein2_left_state_str_list],
                        catalyst_str=None,
                        catalyst_state_str_list=[],
                        complex_str=controlled_interactions_raw[curr_controlled][const_protein1_right_str] if const_protein1_right_str in controlled_interactions_raw[curr_controlled] else '',
                        complex_state_str_list=controlled_interactions_raw[curr_controlled][const_protein1_right_state_str_list] if const_protein1_right_state_str_list in controlled_interactions_raw[curr_controlled] else '',
                        weight=1,
                        complex_interaction_str=controlled_interactions_raw[curr_controlled][const_interaction_type]
                    )
                complex_type_interaction_obj.protein_1_type \
                    = controlled_interactions_raw[curr_controlled][const_protein1_left_type]
                complex_type_interaction_obj.protein_1_id \
                    = controlled_interactions_raw[curr_controlled][const_protein1_left_id]
                complex_type_interaction_obj.protein_2_type \
                    = controlled_interactions_raw[curr_controlled][const_protein2_left_type]
                complex_type_interaction_obj.protein_2_id \
                    = controlled_interactions_raw[curr_controlled][const_protein2_left_id]
                complex_type_interaction_obj.complex_type \
                    = controlled_interactions_raw[curr_controlled][const_protein1_right_type] \
                    if const_protein1_right_type in controlled_interactions_raw[curr_controlled] else ''
                complex_type_interaction_obj.complex_id\
                    = controlled_interactions_raw[curr_controlled][const_protein1_right_id] \
                    if const_protein1_right_id in controlled_interactions_raw[curr_controlled] else ''
            else:
                complex_type_interaction_obj =\
                    i.ComplexTypeInteraction(
                        protein_1_str=controlled_interactions_raw[curr_controlled][const_protein1_right_str],
                        protein_1_state_str_list=controlled_interactions_raw[curr_controlled][const_protein1_right_state_str_list],
                        protein_2_str=controlled_interactions_raw[curr_controlled][const_protein2_right_str],
                        protein_2_state_str_list=controlled_interactions_raw[curr_controlled][const_protein2_right_state_str_list],
                        catalyst_str=None,
                        catalyst_state_str_list=[],
                        complex_str=controlled_interactions_raw[curr_controlled][const_protein1_left_str],
                        complex_state_str_list=controlled_interactions_raw[curr_controlled][const_protein1_left_state_str_list],
                        weight=1,
                        complex_interaction_str=controlled_interactions_raw[curr_controlled][const_interaction_type]
                    )
                complex_type_interaction_obj.protein_1_type \
                    = controlled_interactions_raw[curr_controlled][const_protein1_right_type]
                complex_type_interaction_obj.protein_1_id \
                    = controlled_interactions_raw[curr_controlled][const_protein1_right_id]
                complex_type_interaction_obj.protein_2_type \
                    = controlled_interactions_raw[curr_controlled][const_protein2_right_type]
                complex_type_interaction_obj.protein_2_id \
                    = controlled_interactions_raw[curr_controlled][const_protein2_right_id]
                complex_type_interaction_obj.complex_type \
                    = controlled_interactions_raw[curr_controlled][const_protein1_left_type]
                complex_type_interaction_obj.complex_id\
                    = controlled_interactions_raw[curr_controlled][const_protein1_left_id]
            #
            complex_type_interaction_obj.is_left_to_right = is_left_to_right
        else:
            if const_protein1_right_str not in controlled_interactions_raw[curr_controlled]:
                if not is_skip:
                    raise AssertionError
                else:
                    continue
            if debug:
                print 'controlled_interactions_raw[curr_controlled][const_protein1_right_state_str_list]', controlled_interactions_raw[curr_controlled][const_protein1_right_state_str_list]
            interaction_obj =\
                i.Interaction(
                    protein_str=controlled_interactions_raw[curr_controlled][const_protein1_left_str],
                    protein_state_str_list=controlled_interactions_raw[curr_controlled][const_protein1_left_state_str_list],
                    catalyst_str=None,
                    catalyst_state_str_list=[],
                    result_state_str_list=controlled_interactions_raw[curr_controlled][const_protein1_right_state_str_list],
                    weight=1,
                    interaction_str=controlled_interactions_raw[curr_controlled][const_interaction_type]
                )
            if interaction_obj.protein_str != controlled_interactions_raw[curr_controlled][const_protein1_right_str]:
                interaction_obj.protein_result_str = controlled_interactions_raw[curr_controlled][const_protein1_right_str]
                interaction_obj.protein_result_type = controlled_interactions_raw[curr_controlled][const_protein1_right_type]
                interaction_obj.protein_result_id = controlled_interactions_raw[curr_controlled][const_protein1_right_id]
            interaction_obj.protein_type \
                = controlled_interactions_raw[curr_controlled][const_protein1_left_type]
            interaction_obj.protein_id \
                = controlled_interactions_raw[curr_controlled][const_protein1_left_id]
            if is_add_interaction_type_to_result_state:
                interaction_obj.result_state_str_list.append(interaction_obj.interaction_str)
        #
        if is_complex_type:
            curr_obj = complex_type_interaction_obj
            complex_type_interaction_obj = None
        else:
            curr_obj = interaction_obj
            interaction_obj = None
        #
        curr_obj.text_sentence = controlled_interactions_raw[curr_controlled][const_sentence]
        curr_obj.other_type = controlled_interactions_raw[curr_controlled][const_biopax_type]
        curr_obj.other_source_id = curr_controlled
        print 'curr_obj', curr_obj
        #
        if curr_controlled not in controller_control_map:
            interactions_list.append(curr_obj)
        else:
            num_controllers = len(controller_control_map[curr_controlled])
            for curr_idx in range(num_controllers):
                curr_control = controller_control_map[curr_controlled][curr_idx]
                curr_new_obj = copy.deepcopy(curr_obj)
                print 'curr_new_obj', curr_new_obj
                add_control_info_to_interaction_obj(curr_new_obj, curr_control)
                print 'curr_new_obj', curr_new_obj
                interactions_list.append(curr_new_obj)
    print 'Here is the final list of model interactions ...'
    for curr_obj in interactions_list:
        print curr_obj
    print 'Count of final interactions is ', len(interactions_list)
    return interactions_list


def get_interactions_objs_map_frm_list(interactions_list):
    interactions_objs_map = {}
    interactions_objs_map['state_change'] = []
    interactions_objs_map['complex'] = []
    for curr_interaction_obj in interactions_list:
        if isinstance(curr_interaction_obj, i.Interaction):
            interactions_objs_map['state_change'].append(curr_interaction_obj)
        elif isinstance(curr_interaction_obj, i.ComplexTypeInteraction):
            interactions_objs_map['complex'].append(curr_interaction_obj)
        else:
            raise AssertionError
    return interactions_objs_map


def get_protein_name_id_map(interactions_list):
    def add_name_id_to_map(name,id, proteins_map):
        if name not in proteins_map:
            proteins_map[name] = []
        if id is not None and id.strip():
            proteins_map[name] += id.split(',')
            proteins_map[name] = list(set(proteins_map[name]))

    proteins_map = {}
    for curr_obj in interactions_list:
        if curr_obj.catalyst_str is not None:
            add_name_id_to_map(curr_obj.catalyst_str, curr_obj.catalyst_id, proteins_map)
        if isinstance(curr_obj, i.Interaction):
            if curr_obj.protein_str is None:
                raise AssertionError
            add_name_id_to_map(curr_obj.protein_str, curr_obj.protein_id, proteins_map)
        else:
            if curr_obj.protein_1_str is None or curr_obj.protein_2_str is None:
                raise AssertionError
            add_name_id_to_map(curr_obj.protein_1_str, curr_obj.protein_1_id, proteins_map)
            add_name_id_to_map(curr_obj.protein_2_str, curr_obj.protein_2_id, proteins_map)
            if curr_obj.complex_str is not None:
                add_name_id_to_map(curr_obj.complex_str, curr_obj.complex_id, proteins_map)
    return proteins_map


def preprocess_interaction_objs_parse_frm_biopax_csv(interactions_objs_list):
    def add_state_if_no_state_change(interactions_objs_list):
        for curr_interaction_obj in interactions_objs_list:
            if isinstance(curr_interaction_obj, i.Interaction):
                state_diff_list = list(set(curr_interaction_obj.result_state_str_list) - set(curr_interaction_obj.protein_state_str_list))
                if not state_diff_list:
                    print 'curr_interaction_obj', curr_interaction_obj
                    if 'template' in curr_interaction_obj.other_type.lower():
                        if 'transcr' in curr_interaction_obj.text_sentence.lower():
                            curr_interaction_obj.result_state_str_list.append('transcription')
                        elif 'expres' in curr_interaction_obj.text_sentence.lower():
                            curr_interaction_obj.result_state_str_list.append('expression')
                        elif 'prod' in curr_interaction_obj.text_sentence.lower():
                            curr_interaction_obj.result_state_str_list.append('production')
                        elif curr_interaction_obj.text_sentence is None or not curr_interaction_obj.text_sentence.strip():
                            curr_interaction_obj.result_state_str_list.append('increase')
                        else:
                            raise NotImplementedError
                    elif 'conversion' in curr_interaction_obj.other_type.lower():
                        continue
                    elif 'biochemical' in curr_interaction_obj.other_type.lower():
                        continue
                    elif 'degrad' in curr_interaction_obj.other_type.lower():
                        curr_interaction_obj.result_state_str_list.append('degraded')
                    else:
                        raise NotImplementedError

    def process_identifiers(interactions_objs_list):
        def get_id(curr_id, curr_name_str):
            curr_id = extract_id_frm_identifier_url(curr_id)
            if curr_id is None:
                curr_id = ''
            #
            curr_id_darpa = imoo.iowl_obj.get_identifier(curr_name_str)
            if curr_id_darpa is None or (not curr_id_darpa.strip()):
                curr_id_darpa = pido.pid_obj.get_identifier(curr_name_str)
            if curr_id_darpa is None:
                curr_id_darpa = ''
            #
            if curr_id.strip() and curr_id_darpa.strip():
                curr_id = curr_id_darpa + ',' + curr_id
            elif curr_id_darpa.strip():
                curr_id = curr_id_darpa
            curr_id.strip(',')
            if curr_id.strip():
                return ','.join(list(set(curr_id.split(','))))

        for curr_obj in interactions_objs_list:
            curr_obj.catalyst_id = get_id(curr_obj.catalyst_id, curr_obj.catalyst_str)
            if isinstance(curr_obj, i.Interaction):
                curr_obj.protein_id = get_id(curr_obj.protein_id, curr_obj.protein_str)
            elif isinstance(curr_obj, i.ComplexTypeInteraction):
                curr_obj.protein_1_id = get_id(curr_obj.protein_1_id, curr_obj.protein_1_str)
                curr_obj.protein_2_id = get_id(curr_obj.protein_2_id, curr_obj.protein_2_str)
                curr_obj.complex_id = get_id(curr_obj.complex_id, curr_obj.complex_str)

    #as function name suggest, it is specifically for pre-processing interactions objs parsed from biopax csv (or biopax model in general)
    print 'preprocessing interaction objects from biopax csv'
    add_state_if_no_state_change(interactions_objs_list)
    process_identifiers(interactions_objs_list)


class BioPAXModel:
    def __init__(self, is_map=True, is_protein_name_id_map=True):
        def set_model_identifiers_list():
            if self.protein_identifier_list_map is not None:
                # self.identifiers_list = []
                # for curr_name in self.protein_identifier_list_map:
                #     self.identifiers_list += self.protein_identifier_list_map[curr_name]
                self.identifiers_list = ioo.iowl_obj.identifiers_list
                self.identifiers_list = list(set(self.identifiers_list))
                if not ch.is_hpcc:
                    with open(cap.absolute_path+'../identifiers_list.json', 'w') as f:
                        json.dump(self.identifiers_list, f, ensure_ascii=True, sort_keys=True, indent=5)
            else:
                self.identifiers_list = None

        def set_protein_identifier_list_map():
            if is_protein_name_id_map:
                self.protein_identifier_list_map = get_protein_name_id_map(self.interactions_objs_list)
                set_model_identifiers_list()
                if not ch.is_hpcc:
                    with open(cap.absolute_path+'../protein_identifier_list_map.json', 'w') as f:
                        json.dump(self.protein_identifier_list_map, f, ensure_ascii=True, sort_keys=True, indent=5)
            else:
                self.protein_identifier_list_map = None

        def set_json_objs():
            start_time = time.time()
            ija_obj = ija.Interaction_to_JSON(is_model=True)
            model_json_objs = ija_obj.interaction_obj_to_darpa_json(self.interactions_objs_list,
                                                                    protein_name_idlist_map=self.protein_identifier_list_map)
            self.json_objs = model_json_objs
            print 'Number of interactions objects in DARPA json format are ', len(self.json_objs)
            if not ch.is_hpcc:
                ija_obj.write_json_index_cards(model_json_objs)
            print 'Time to generate the model index cards was ', time.time()-start_time

        def set_interactions_objs_map():
            if is_map:
                self.interactions_objs_map = get_interactions_objs_map_frm_list(self.interactions_objs_list)
            else:
                self.interactions_objs_map = None

        control_interactions_raw, controller_control_map = parse_controller()
        controlled_interactions_raw = parse_controlled()
        interactions_objs_list = get_interactions_objs_from_raw(control_interactions_raw, controller_control_map,
                                                                controlled_interactions_raw)
        preprocess_interaction_objs_parse_frm_biopax_csv(interactions_objs_list)
        self.interactions_objs_list = interactions_objs_list
        print 'Number of interactions objs in ISI format are ', len(self.interactions_objs_list)
        #
        set_protein_identifier_list_map()
        #
        set_interactions_objs_map()
        #
        set_json_objs()


if __name__ == '__main__':
    biopax_model_obj = BioPAXModel()
