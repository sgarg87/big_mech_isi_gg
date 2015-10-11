import copy
import config
import model_relation_types as mrt
import interactions as i
from config_console_output import *


state_change = 'state_change'
complex = 'complex'


def add_corroboration_conflict_specialization_information(model_interactions_list, curr_obj):
    try:
        corroborating_model_obj_idx = model_interactions_list.index(curr_obj)
        curr_obj.model_relation_map[mrt.Corroboration] = model_interactions_list[corroborating_model_obj_idx]
    except ValueError:
        pass
    #
    curr_conflict_obj = copy.deepcopy(curr_obj)
    curr_conflict_obj.is_positive_catalyst = not curr_conflict_obj.is_positive_catalyst
    try:
        conflicting_model_obj_idx = model_interactions_list.index(curr_conflict_obj)
        curr_obj.model_relation_map[mrt.Conflicting] = model_interactions_list[conflicting_model_obj_idx]
    except ValueError:
        pass
    #
    curr_obj_no_catalyst = copy.deepcopy(curr_obj)
    curr_obj_no_catalyst.catalyst_str = None
    curr_obj_no_catalyst.catalyst_state_str_list = []
    try:
        specialization_model_obj_idx = model_interactions_list.index(curr_obj_no_catalyst)
        curr_obj.model_relation_map[mrt.Specialization] = model_interactions_list[specialization_model_obj_idx]
    except ValueError:
        pass

def add_model_link(model_protein_names, curr_obj):
    if model_protein_names is not None and model_protein_names:
        link_list = ''
        if curr_obj.catalyst_str is not None and curr_obj.catalyst_str.lower() in model_protein_names:
            link_list += curr_obj.catalyst_str
            link_list += ','
        if isinstance(curr_obj, i.ComplexTypeInteraction):
            if curr_obj.protein_1_str.lower() in model_protein_names:
                link_list += curr_obj.protein_1_str
                link_list += ','
            if curr_obj.protein_2_str.lower() in model_protein_names:
                link_list += curr_obj.protein_2_str
                link_list += ','
            if curr_obj.complex_str is not None and curr_obj.complex_str.lower() in model_protein_names:
                link_list += curr_obj.complex_str
                link_list += ','
        elif isinstance(curr_obj, i.Interaction):
            if curr_obj.protein_str.lower() in model_protein_names:
                link_list += curr_obj.protein_str
                link_list += ','
        else:
            raise AssertionError
        link_list = link_list.rstrip(',')
    else:
        link_list = ''
    curr_obj.protein_comma_list_link_to_model = link_list


def add_read_against_model_info_to_interaction_objs(state_change_objs_list, model_interactions_map, model_protein_names):
    if model_protein_names is not None and model_protein_names:
        model_protein_names = [x.lower() for x in model_protein_names]
    for curr_state_change_obj in state_change_objs_list:
        if state_change in model_interactions_map and model_interactions_map[state_change]:
            add_corroboration_conflict_specialization_information(model_interactions_map[state_change], curr_state_change_obj)
        if model_protein_names is not None and model_protein_names:
            add_model_link(model_protein_names, curr_state_change_obj)


def add_read_against_model_info_to_complex_form_type_objs(complex_objs, model_interactions, model_protein_names):
    if model_protein_names is not None and model_protein_names:
        model_protein_names = [x.lower() for x in model_protein_names]
    #X is an array of protein state changes
    for curr_complex_form_obj in complex_objs:
        if complex in model_interactions and model_interactions[complex]:
            add_corroboration_conflict_specialization_information(model_interactions[complex], curr_complex_form_obj)
        if model_protein_names is not None and model_protein_names:
            add_model_link(model_protein_names, curr_complex_form_obj)
