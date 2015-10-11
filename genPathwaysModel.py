import numpy.random as rnd
import numpy as np
from constants import *
import pickle
import heapq
import ntpath
import itertools
import sparray as sa
reload(sa)
from interactions import *
import copy
import csv
import config_gen_pathways_model as cgp
import parse_model_frm_csv_biopax as pmcb
import constants_absolute_path as cap
from config_console_output import *


interaction_prior_value = 0.05
min_posterior_lkl = interaction_prior_value*1.02
# min_posterior_lkl = interaction_prior_value*2
max_num_inference_state_change = 250
max_num_inference_complex_forms = 150

num_interactions_frm_amr = 1000

is_sparse = True #always use sparse, many changes made specifically considering sparse representation. non-sparse may not provide full functionality or may not even work


def read_sentence(amr_dot_file_path):
    sentence = ''
    with open(cap.absolute_path+amr_dot_file_path + '_s', 'r') as f:
        for curr_line in f:
            sentence += curr_line
    return sentence


def load_interactions(amr_dot_file_path):
    f = open(cap.absolute_path+amr_dot_file_path + '_ei', 'rb')
    interactions = pickle.load(f)
    f.close()
    return interactions


def main(file_path='./'):
    raise DeprecationWarning
    protein_names = init_proteins_set()
    state_names = init_states_set()
    phi_ijkl, phi_i_jk = generate_noise_model(protein_names.size-2)
    X = init_ground_truth_protein_state_changes()
    write_protein_state_changes(X, protein_names, state_names, file_path+'ground_truth.txt')
    Y = init_noisy_protein_state_changes()
    write_protein_state_changes(Y, protein_names, state_names, file_path+'noisy_observations.txt')
    psi_i_j = init_proteins_state_change_model(X)
    X_inf = infer_interactions(Y, psi_i_j, phi_ijkl, phi_i_jk)
    write_protein_state_changes(X_inf, protein_names, state_names, file_path+'inferred.txt')


def main_fr_amr(interactions_amr_dot_file_paths, amr_dot_files_path=None, is_infer=True, model_interactions=None, passage_name=None, result_file_path=None, model_output_file_path=None, protein_identifier_map=None, protein_family_map=None):
    protein_names_joint = np.array([], np.object)
    state_names_joint = np.array([], np.object)
    count = 0
    org_state_changes_count = 0
    org_complex_forms_count = 0
    psi_i_j = None
    psi_i_jk_l = None
    if is_infer:
        state_changes_inf = None
        complex_forms_inf = None
    if model_interactions is not None:
        count += 1
        protein_names_joint, state_names_joint = get_protein_and_state_names_from_interactions_list(model_interactions)
        model_protein_names = protein_names_joint.tolist()
        n = protein_names_joint.size-2 #no. of proteins excluding None and Garbage
        k = state_names_joint.size #no. of state bits
        m = 2**k #no of states
        model_state_changes, _, model_complex_forms, _ = get_protein_state_changes_and_complex_from_interactions(model_interactions, protein_names_joint, state_names_joint)
        if model_output_file_path is not None:
            append_flag = False
            if model_state_changes is not None:
                write_protein_state_changes(model_state_changes, protein_names_joint, state_names_joint, model_output_file_path+'.txt', is_append=append_flag, is_model=True, model_protein_names=model_protein_names)
                write_protein_state_changes(model_state_changes, protein_names_joint, state_names_joint, model_output_file_path+'.csv', is_append=append_flag, is_csv=True, is_model=True, model_protein_names=model_protein_names)
                write_json_index_cards_protein_state_changes(model_state_changes, protein_names_joint, state_names_joint, model_output_file_path+'.json', is_append=append_flag, is_model=True, model_protein_names=model_protein_names)
                append_flag = True
            if model_complex_forms is not None:
                write_protein_complex_formations(model_complex_forms, protein_names_joint, state_names_joint, model_output_file_path+'.txt', is_append=append_flag, is_model=True, model_protein_names=model_protein_names)
                write_protein_complex_formations(model_complex_forms, protein_names_joint, state_names_joint, model_output_file_path+'.csv', is_append=append_flag, is_csv=True, is_model=True, model_protein_names=model_protein_names)
                write_json_index_cards_protein_complex_formations(model_complex_forms, protein_names_joint, state_names_joint, model_output_file_path+'.json', is_append=append_flag, is_model=True, model_protein_names=model_protein_names)
    else:
        model_state_changes = None
        model_complex_forms = None
        model_protein_names = None
    sentences = []
    for interactions_amr_dot_file_path in interactions_amr_dot_file_paths:
        print 'interactions_amr_dot_file_path:', interactions_amr_dot_file_path
        count += 1
        interactions = load_interactions(interactions_amr_dot_file_path)
        sentence = read_sentence(interactions_amr_dot_file_path)
        if sentence:
            sentences.append(sentence)
        protein_names, state_names = get_protein_and_state_names_from_interactions_list(interactions)
        if count == 1:
            protein_names_joint = protein_names
            del protein_names
            state_names_joint = state_names
            del state_names
        elif count > 1:
            new_protein_names = np.setdiff1d(protein_names, protein_names_joint)
            del protein_names
            new_state_names = np.setdiff1d(state_names, state_names_joint)
            del state_names
            #sequence of concatenation is important
            #do not sort here
            if (new_protein_names is not None) and (new_protein_names.size > 0):
                #order is very important here, first NULL, then previous states except garbage, and then new states, then garbage
                protein_names_joint = np.concatenate((protein_names_joint[:-1], new_protein_names, np.array([protein_names_joint[-1]])))
            del new_protein_names
            #sequence of concatenation is important
            #do not sort here
            if (new_state_names is not None) and (new_state_names.size > 0):
                state_names_joint = np.concatenate((new_state_names, state_names_joint)) #order is very important here
            del new_state_names
        else:
            raise AssertionError
        if protein_names_joint.size < 2:
            raise AssertionError
        # elif protein_names_joint.size == 2:
        #     print 'No protein involved in interactions. Returning from the function ...'
        #     return
        print 'protein_names: ', protein_names_joint
        print 'state_names: ', state_names_joint
        n = protein_names_joint.size-2 #no. of proteins excluding None and Garbage
        k = state_names_joint.size #no. of state bits
        m = 2**k #no of states
        # phi_ijkl, phi_i_jk = generate_noise_model(n)
        # phi_s = generate_noise_state_model(m)
        state_changes, state_change_weights, complex_forms, complex_form_weights = get_protein_state_changes_and_complex_from_interactions(interactions, protein_names_joint, state_names_joint)
        if state_changes is not None:
            write_protein_state_changes(state_changes, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_evidence.txt', is_append=False, model_interactions=model_interactions, X_lkl=state_change_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
            write_json_index_cards_protein_state_changes(state_changes, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_evidence.json', is_append=False, model_interactions=model_interactions, X_lkl=state_change_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
            state_changes = state_changes[0:min(state_changes.shape[0], num_interactions_frm_amr), :]
            state_change_weights = state_change_weights[0:min(state_changes.shape[0], num_interactions_frm_amr)]
            org_state_changes_count += state_changes.shape[0]
            write_protein_state_changes(state_changes, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_evidence_most_lkl.txt', is_append=False, model_interactions=model_interactions, X_lkl=state_change_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
            write_json_index_cards_protein_state_changes(state_changes, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_evidence_most_lkl.json', is_append=False, model_interactions=model_interactions, X_lkl=state_change_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
            if is_infer:
                if psi_i_j is None:
                    if not is_sparse:
                        psi_i_j = init_gen_proteins_state_change_model(n, k)
                    else:
                        psi_i_j = init_gen_proteins_state_change_model_sparse(n, k, model_state_changes)
                        psi_i_j_init = copy.deepcopy(psi_i_j)
                else:
                    if not is_sparse:
                        psi_i_j = extend_prior_model(n, k, psi_i_j)
                    else:
                        psi_i_j = extend_prior_model_sparse(n, k, psi_i_j)
                #psi is updated inside this function
                infer_interactions(state_changes, state_change_weights, psi_i_j, None, None, None, protein_names_joint, state_names_joint)
        if complex_forms is not None:
            if state_changes is not None:
                append_flag = True
            else:
                append_flag = False
            print 'append_flag:', append_flag
            print 'complex_forms:', complex_forms
            print 'protein_names_joint:', protein_names_joint
            print 'state_names_joint:', state_names_joint
            write_protein_complex_formations(complex_forms, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_evidence.txt', is_append=append_flag, model_interactions=model_interactions, X_lkl=complex_form_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
            write_json_index_cards_protein_complex_formations(complex_forms, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_evidence.json', is_append=append_flag, model_interactions=model_interactions, X_lkl=complex_form_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
            complex_forms = complex_forms[0:min(complex_forms.shape[0], num_interactions_frm_amr), :]
            complex_form_weights = complex_form_weights[0:min(complex_forms.shape[0], num_interactions_frm_amr)]
            org_complex_forms_count += complex_forms.shape[0]
            write_protein_complex_formations(complex_forms, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_evidence_most_lkl.txt', is_append=append_flag, model_interactions=model_interactions, X_lkl=complex_form_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
            write_json_index_cards_protein_complex_formations(complex_forms, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_evidence_most_lkl.json', is_append=append_flag, model_interactions=model_interactions, X_lkl=complex_form_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
            if is_infer:
                if psi_i_jk_l is None:
                    if not is_sparse:
                        psi_i_jk_l = init_gen_proteins_complex_model(n, k)
                    else:
                        psi_i_jk_l = init_gen_proteins_complex_model_sparse(n, k, model_complex_forms)
                        psi_i_jk_l_init = copy.deepcopy(psi_i_jk_l)
                else:
                    if not is_sparse:
                        psi_i_jk_l = extend_prior_model(n, k, psi_i_jk_l)
                    else:
                        psi_i_jk_l = extend_prior_model_sparse(n, k, psi_i_jk_l)
                #psi is updated inside this function
                infer_interactions(complex_forms, complex_form_weights, psi_i_jk_l, None, None, None, protein_names_joint, state_names_joint)
        #writing inference in sequence after processing an AMR (posterior for one AMR becomes prior for next)
        #state change
        if is_infer:
            state_changes_inf_subset = None
            if psi_i_j is not None:
                if not is_sparse:
                    state_changes_inf, state_changes_inf_lkl = get_most_lkl_interactions(psi_i_j)
                else:
                    state_changes_inf, state_changes_inf_lkl = get_most_lkl_interactions_sparse(psi_i_j, psi_i_j_init, protein_names_joint, state_names_joint)
                print 'state_changes_inf:', state_changes_inf
                print 'state_changes_inf_lkl:', state_changes_inf_lkl
                write_protein_state_changes(state_changes_inf, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.txt', state_changes_inf_lkl, is_append=False, amr_file=interactions_amr_dot_file_path, model_interactions=model_interactions, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
                write_json_index_cards_protein_state_changes(state_changes_inf, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.json', state_changes_inf_lkl, is_append=False, amr_file=interactions_amr_dot_file_path, model_interactions=model_interactions, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
                state_changes_inf_subset = (state_changes_inf if state_changes_inf.shape[0] <= max_num_inference_state_change else state_changes_inf[0:max_num_inference_state_change])
                # write_protein_state_changes_sif(state_changes_inf_subset, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.sif', state_changes_inf_lkl, is_append=False, amr_file=interactions_amr_dot_file_path)
            #complex
            complex_forms_inf_subset = None
            if psi_i_jk_l is not None:
                if not is_sparse:
                    complex_forms_inf, complex_forms_inf_lkl = get_most_lkl_interactions(psi_i_jk_l)
                else:
                    complex_forms_inf, complex_forms_inf_lkl = get_most_lkl_interactions_sparse(psi_i_jk_l, psi_i_jk_l_init, protein_names_joint, state_names_joint)
                print 'complex_forms_inf:', complex_forms_inf
                print 'complex_forms_inf_lkl:', complex_forms_inf_lkl
                write_protein_complex_formations(complex_forms_inf, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.txt', complex_forms_inf_lkl, is_append=True, amr_file=interactions_amr_dot_file_path, model_interactions=model_interactions, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
                write_json_index_cards_protein_complex_formations(complex_forms_inf, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.json', complex_forms_inf_lkl, is_append=True, amr_file=interactions_amr_dot_file_path, model_interactions=model_interactions, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
                complex_forms_inf_subset = (complex_forms_inf if complex_forms_inf.shape[0] <= max_num_inference_complex_forms else complex_forms_inf[0:max_num_inference_complex_forms])
                # write_protein_complex_formations_sif(complex_forms_inf_subset, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.sif', complex_forms_inf_lkl, is_append=True, amr_file=interactions_amr_dot_file_path)
            if state_changes_inf_subset is not None or complex_forms_inf_subset is not None:
                # # write owl
                # write_owl(state_changes_inf_subset, complex_forms_inf_subset, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.owl')
                #add evidence to inferred interactions file in the end
                f = open(cap.absolute_path+interactions_amr_dot_file_path+'_inferred.txt', 'a')
                f.write('\n\n\n###############################################################################')
                f.write('\n###############################################################################')
                f.write('Evidence interactions extracted from this current AMR to extend the model follow.')
                f.close()
            if state_changes is not None:
                write_protein_state_changes(state_changes, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.txt', is_append=True, amr_file=interactions_amr_dot_file_path, model_interactions=model_interactions, X_lkl=state_change_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
                write_json_index_cards_protein_state_changes(state_changes, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.json', is_append=True, amr_file=interactions_amr_dot_file_path, model_interactions=model_interactions, X_lkl=state_change_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
            if complex_forms is not None:
                write_protein_complex_formations(complex_forms, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.txt', is_append=True, amr_file=interactions_amr_dot_file_path, model_interactions=model_interactions, X_lkl=complex_form_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
                write_json_index_cards_protein_complex_formations(complex_forms, protein_names_joint, state_names_joint, interactions_amr_dot_file_path+'_inferred.json', is_append=True, amr_file=interactions_amr_dot_file_path, model_interactions=model_interactions, X_lkl=complex_form_weights, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
    if is_infer:
        try:
            if state_changes_inf is not None or complex_forms_inf is not None:
                append_flag = False
                if state_changes_inf is not None:
                    write_protein_state_changes(state_changes_inf, protein_names_joint, state_names_joint, result_file_path+'.txt', is_append=append_flag, amr_file=None, model_interactions=model_interactions, X_lkl=state_changes_inf_lkl, passage_name=passage_name, is_csv=False, model_protein_names=model_protein_names, sentences=sentences)
                    write_protein_state_changes(state_changes_inf, protein_names_joint, state_names_joint, result_file_path+'.csv', is_append=append_flag, amr_file=None, model_interactions=model_interactions, X_lkl=state_changes_inf_lkl, passage_name=passage_name, is_csv=True, model_protein_names=model_protein_names, sentences=sentences)
                    write_json_index_cards_protein_state_changes(state_changes_inf, protein_names_joint, state_names_joint, result_file_path+'.json', is_append=append_flag, amr_file=None, model_interactions=model_interactions, X_lkl=state_changes_inf_lkl, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
                    append_flag = True
                if complex_forms_inf is not None:
                    write_protein_complex_formations(complex_forms_inf, protein_names_joint, state_names_joint, result_file_path+'.txt', is_append=append_flag, amr_file=None, model_interactions=model_interactions, X_lkl=complex_forms_inf_lkl, passage_name=passage_name, is_csv=False, model_protein_names=model_protein_names, sentences=sentences)
                    write_protein_complex_formations(complex_forms_inf, protein_names_joint, state_names_joint, result_file_path+'.csv', is_append=append_flag, amr_file=None, model_interactions=model_interactions, X_lkl=complex_forms_inf_lkl, passage_name=passage_name, is_csv=True, model_protein_names=model_protein_names, sentences=sentences)
                    write_json_index_cards_protein_complex_formations(complex_forms_inf, protein_names_joint, state_names_joint, result_file_path+'.json', is_append=append_flag, amr_file=None, model_interactions=model_interactions, X_lkl=complex_forms_inf_lkl, passage_name=passage_name, model_protein_names=model_protein_names, sentences=sentences)
        except NameError as e:
            print e
    if protein_identifier_map is not None and protein_family_map is not None:
        write_proteins_info(protein_names_joint.tolist()[1:-1], protein_identifier_map, protein_family_map, result_file_path + '_entities' + '.csv')


def write_proteins_info(protein_names, protein_identifier_map, protein_family_map, file_path):
    #X is an array of protein state changes
    f = open(cap.absolute_path+file_path, 'w')
    f.write('Entity,Identifier,Family')
    f.write('\n')
    for curr_protein_name in protein_names:
        if curr_protein_name is not None and curr_protein_name:
            if curr_protein_name in protein_identifier_map:
                curr_protein_identifier = protein_identifier_map[curr_protein_name]
            else:
                curr_protein_identifier = ''
            if curr_protein_name in protein_family_map:
                curr_protein_family = protein_family_map[curr_protein_name]
            else:
                curr_protein_family = ''
            f.write(curr_protein_name+','+curr_protein_identifier+','+curr_protein_family)
            f.write('\n')
    f.close()


def extend_prior_model(n, k, psi):
    #n is no. of protein excluding NULL, GARBAGE
    #k is no of state bits
    print 'extending prior model ...'
    m = 2**k #no. of states
    psi_shape = psi.shape
    if psi_shape[0] == n+1 and psi_shape[1] == m:
        print 'returning same prior model since no extension to make'
        return psi
    #state change
    if len(psi_shape) == state_change_dim:
        psi_new = init_gen_proteins_state_change_model(n, k)
        mesh_idx = np.meshgrid(np.arange(psi_shape[0]), np.arange(psi_shape[1]), np.arange(psi_shape[2]), np.arange(psi_shape[3]), np.arange(psi_shape[4]), np.arange(psi_shape[5]), indexing='ij')
        psi_new[mesh_idx] = psi
        return psi_new
    elif len(psi_shape) == complex_form_dim:
        psi_new = init_gen_proteins_complex_model(n, k)
        mesh_idx = np.meshgrid(np.arange(psi_shape[0]), np.arange(psi_shape[1]), np.arange(psi_shape[2]), np.arange(psi_shape[3]), np.arange(psi_shape[4]), np.arange(psi_shape[5]), np.arange(psi_shape[6]), np.arange(psi_shape[7]), np.arange(psi_shape[8]), indexing='ij')
        psi_new[mesh_idx] = psi
        return psi_new
    else:
        raise AssertionError


def extend_prior_model_sparse(n, k, psi):
    #n is no. of protein excluding NULL, GARBAGE
    #k is no of state bits
    print 'extending prior model ...'
    m = 2**k #no. of states
    psi_shape = psi.shape
    if psi_shape[0] == n+1 and psi_shape[1] == m:
        print 'returning same prior model since no extension to make'
        return psi
    #state change
    if len(psi_shape) == state_change_dim:
        psi_new = init_gen_proteins_state_change_model_sparse(n, k)
    elif len(psi_shape) == complex_form_dim:
        psi_new = init_gen_proteins_complex_model_sparse(n, k)
    else:
        raise AssertionError
    psi_new[psi.data.keys()] = psi
    return psi_new


def get_state_change_interaction_obj(protein_names, state_names, x, lkl=None):
    if isinstance(x, list):
        x = np.array(x)
    k = state_names.size
    if x.size == state_change_dim: #its a state change interaction
        #position i
        if x[0] != 0:
            catalyst_name = protein_names[x[0]]
        else:
            catalyst_name = None
        catalyst_state = get_state_string_vector_from(get_state_vector_from(x[1], k), state_names).tolist()
        #position j
        if x[2] != 0:
            protein_name = protein_names[x[2]]
        else:
            protein_name = None
        protein_state = get_state_string_vector_from(get_state_vector_from(x[3], k), state_names).tolist()
        protein_state_new = get_state_string_vector_from(get_state_vector_from(x[4], k), state_names).tolist()
        if x[5] == 1:
            is_pos_cat = True
        elif x[5] == 0:
            is_pos_cat = False
        else:
            raise AssertionError
        interaction = Interaction(protein_name, protein_state, catalyst_name, catalyst_state, protein_state_new, None, is_positive_catalyst=is_pos_cat, weight=(1 if lkl is None else lkl))
    else:
        raise AssertionError
    return interaction


def get_complex_form_obj(protein_names, state_names, x, lkl=None):
    if isinstance(x, list):
        x = np.array(x)
    k = state_names.size
    if x.size == complex_form_dim: #complex formation interaction
        if x[0] != 0:
            protein_i_name = protein_names[x[0]]
        else:
            protein_i_name = None
        protein_i_state = get_state_string_vector_from(get_state_vector_from(x[1], k), state_names).tolist()
        if x[2] != 0:
            protein_j_name = protein_names[x[2]]
        else:
            protein_j_name = None
        protein_j_state = get_state_string_vector_from(get_state_vector_from(x[3], k), state_names).tolist()
        if x[4] != 0:
            protein_k_name = protein_names[x[4]]
        else:
            protein_k_name = None
        protein_k_state = get_state_string_vector_from(get_state_vector_from(x[5], k), state_names).tolist()
        if x[6] != 0:
            protein_l_name = protein_names[x[6]]
        else:
            protein_l_name = None
        protein_l_state = get_state_string_vector_from(get_state_vector_from(x[7], k), state_names).tolist()
        if x[8] == 1:
            is_pos_cat = True
        elif x[8] == 0:
            is_pos_cat = False
        else:
            raise AssertionError
        complex_form = ComplexTypeInteraction(protein_j_name, protein_j_state, protein_k_name, protein_k_state, protein_i_name, protein_i_state, protein_l_name, protein_l_state, is_positive_catalyst=is_pos_cat, weight=(1 if lkl is None else lkl))
    else:
        raise AssertionError
    return complex_form


def interaction_to_str(protein_names, state_names, x, lkl=None):
    k = state_names.size
    if x.size == 5: #its a state change interaction
        protein_i_name = protein_names[x[0]]
        protein_i_state = get_state_string_vector_from(get_state_vector_from(x[1], k), state_names)
        protein_j_name = protein_names[x[2]]
        protein_j_prv_state = get_state_string_vector_from(get_state_vector_from(x[3], k), state_names)
        protein_j_new_state = get_state_string_vector_from(get_state_vector_from(x[4], k), state_names)
        if x[0] != 0: #catalyst protein is not NULL:
            catalyst_str = '\n\tcatalyst:{}{}'.format(protein_i_name, states_to_str(protein_i_state))
        else:
            catalyst_str = ''
        out_str = '{}{}  ---->  {}{}'.format(protein_j_name, states_to_str(protein_j_prv_state), protein_j_name, states_to_str(protein_j_new_state))
        out_str = '\n{' + ('posterior_lkl:' + str(lkl)+':' if lkl is not None else '') + catalyst_str + '\n\t' + out_str + '\n}'
    elif x.size == 8: #complex formation interaction
        protein_i_name = protein_names[x[0]]
        protein_i_state = get_state_string_vector_from(get_state_vector_from(x[1], k), state_names)
        protein_j_name = protein_names[x[2]]
        protein_j_state = get_state_string_vector_from(get_state_vector_from(x[3], k), state_names)
        protein_k_name = protein_names[x[4]]
        protein_k_state = get_state_string_vector_from(get_state_vector_from(x[5], k), state_names)
        protein_l_name = protein_names[x[6]]
        protein_l_state = get_state_string_vector_from(get_state_vector_from(x[7], k), state_names)
        if x[0] != 0: #catalyst protein is not NULL:
            catalyst_str = '\n\tcatalyst:{}{}'.format(protein_i_name, states_to_str(protein_i_state))
        else:
            catalyst_str = ''
        out_str = '{}{}, {}{}  ---->  {}{}'.format(protein_j_name, states_to_str(protein_j_state), protein_k_name, states_to_str(protein_k_state), protein_l_name, states_to_str(protein_l_state))
        out_str = '\n{' + ('posterior_lkl:' + str(lkl) if lkl is not None else '') + catalyst_str + '\n\t' + out_str + '\n}'
    else:
        raise AssertionError
    return out_str


def interaction_to_eng(protein_names, state_names, x, lkl=None):
    k = state_names.size
    if x.size == 5: #its a state change interaction
        protein_i_name = protein_names[x[0]]
        protein_i_state = get_state_string_vector_from(get_state_vector_from(x[1], k), state_names)
        protein_j_name = protein_names[x[2]]
        protein_j_prv_state = get_state_string_vector_from(get_state_vector_from(x[3], k), state_names)
        protein_j_new_state = get_state_string_vector_from(get_state_vector_from(x[4], k), state_names)
        if x[0] != 0: #catalyst protein is not NULL:
            out_str = '\n{} in state {}'.format(protein_i_name, states_to_str(protein_i_state))
            out_str += ' changes state of {} from {} to {}'.format(protein_j_name, states_to_str(protein_j_prv_state), states_to_str(protein_j_new_state))
        else:
            out_str = 'State of {} is changed from {} to {}'.format(protein_j_name, states_to_str(protein_j_prv_state), states_to_str(protein_j_new_state))
    elif x.size == 8: #complex formation interaction
        protein_i_name = protein_names[x[0]]
        protein_i_state = get_state_string_vector_from(get_state_vector_from(x[1], k), state_names)
        protein_j_name = protein_names[x[2]]
        protein_j_state = get_state_string_vector_from(get_state_vector_from(x[3], k), state_names)
        protein_k_name = protein_names[x[4]]
        protein_k_state = get_state_string_vector_from(get_state_vector_from(x[5], k), state_names)
        protein_l_name = protein_names[x[6]]
        protein_l_state = get_state_string_vector_from(get_state_vector_from(x[7], k), state_names)
        if x[0] != 0: #catalyst protein is not NULL:
            out_str = '\nWith {} in state {}, '.format(protein_i_name, states_to_str(protein_i_state))
        else:
            out_str = ''
        out_str += '{} in state {} binds with {} in state {} to form new complex {} in state {}'.format(protein_j_name, states_to_str(protein_j_state), protein_k_name, states_to_str(protein_k_state), protein_l_name, states_to_str(protein_l_state))
    else:
        raise AssertionError
    return out_str


def states_to_str(states):
    states_str = ','.join(map(str, states))
    states_str = '[' + states_str + ']'
    return states_str


def interaction_to_str_sif(protein_names, state_names, x, lkl=None):
    k = state_names.size
    if x.size == 5: #its a state change interaction
        protein_i_name = protein_names[x[0]]
        protein_i_state = get_state_string_vector_from(get_state_vector_from(x[1], k), state_names)
        protein_j_name = protein_names[x[2]]
        protein_j_prv_state = get_state_string_vector_from(get_state_vector_from(x[3], k), state_names)
        protein_j_new_state = get_state_string_vector_from(get_state_vector_from(x[4], k), state_names)
        if inhibit in protein_j_new_state and inhibit not in protein_j_prv_state:
            interaction_type = 'controls-production-of'
        elif phosphorylate in protein_j_new_state and phosphorylate not in protein_j_prv_state:
            interaction_type = 'controls-phosphorylation-of'
        else:
            interaction_type = 'controls-state-change-of'
        if protein_i_state.size == 0:
            protein_i_state = ''
        if protein_j_prv_state.size == 0:
            protein_j_prv_state = ''
        if x[0] != 0: #catalyst protein is not NULL:
            catalyst_str = '{}{}'.format(protein_i_name, states_to_str(protein_i_state))
        else:
            catalyst_str = 'None'
        out_str = '{}\t{}\t{}{}'.format(catalyst_str, interaction_type, protein_j_name, states_to_str(protein_j_prv_state))
    elif x.size == 8: #complex formation interaction
        protein_i_name = protein_names[x[0]]
        protein_i_state = get_state_string_vector_from(get_state_vector_from(x[1], k), state_names)
        protein_j_name = protein_names[x[2]]
        protein_j_state = get_state_string_vector_from(get_state_vector_from(x[3], k), state_names)
        protein_k_name = protein_names[x[4]]
        protein_k_state = get_state_string_vector_from(get_state_vector_from(x[5], k), state_names)
        protein_l_name = protein_names[x[6]]
        protein_l_state = get_state_string_vector_from(get_state_vector_from(x[7], k), state_names)
        if protein_i_state.size == 0:
            protein_i_state = ''
        if protein_j_state.size == 0:
            protein_j_state = ''
        if protein_k_state.size == 0:
            protein_k_state = ''
        if protein_l_state.size == 0:
            protein_l_state = ''
        if x[0] != 0: #catalyst protein is not NULL:
            catalyst_str = '{}{}'.format(protein_i_name, states_to_str(protein_i_state))
        else:
            catalyst_str = 'None'
        catalysis_part = '{}\t{}\t{}{}'.format(catalyst_str, 'controls-production-of', protein_l_name, states_to_str(protein_l_state))
        complex_part = '{}{}\t{}\t{}{}'.format(protein_j_name, protein_j_state, 'in-complex-with', protein_k_name, states_to_str(protein_k_state))
        out_str = catalysis_part + '\n' + complex_part
        # out_str = catalysis_part
    else:
        raise AssertionError
    return out_str


def get_amr_id(amr_dot_file_path):
    return ntpath.basename(amr_dot_file_path.rstrip('.dot'))


def write_json_index_cards_protein_state_changes(X, protein_names, state_names, file_path, X_lkl=None, is_append=False, amr_file=None, model_interactions=None, passage_name=None, is_model=False, model_protein_names=None, sentences=None):
    #X is an array of protein state changes
    state_change_interactions_objs_list = []
    n_x = X.shape[0]
    for curr_idx in np.arange(n_x):
        x = X[curr_idx]
        if X_lkl is not None:
            lkl = X_lkl[curr_idx]
        else:
            lkl = None
        curr_interaction_obj = get_state_change_interaction_obj(protein_names, state_names, x, lkl)
        if model_interactions is not None and not is_model:
            if curr_interaction_obj.catalyst_str is None or not curr_interaction_obj.catalyst_str:
                if curr_interaction_obj.is_mechanistic_information():
                    interaction_relation = 'Extension'
                elif allow_state_change_without_catalyst_nonmechanistic:
                    interaction_relation = 'Extension'
                else:
                    raise AssertionError
            else:
                interaction_relation = 'Extension'
            if state_change in model_interactions and model_interactions[state_change]:
                if curr_interaction_obj in model_interactions[state_change]:
                    interaction_relation = 'Corroboration'
                else:
                    curr_interaction_conflict = copy.deepcopy(curr_interaction_obj)
                    curr_interaction_conflict.is_positive_catalyst = not curr_interaction_conflict.is_positive_catalyst
                    if curr_interaction_conflict in model_interactions[state_change]:
                        interaction_relation = 'Conflicting'
        else:
            interaction_relation = ''
        curr_interaction_obj.model_relation = interaction_relation
        if model_protein_names is not None and model_protein_names:
            link_list = ''
            if curr_interaction_obj.catalyst_str in model_protein_names:
                link_list += curr_interaction_obj.catalyst_str
                link_list += ','
            if curr_interaction_obj.protein_str in model_protein_names:
                link_list += curr_interaction_obj.protein_str
                link_list += ','
            link_list = link_list.rstrip(',')
        else:
            link_list = ''
        curr_interaction_obj.protein_comma_list_link_to_model = link_list
        if sentences is not None:
            source_text = get_relevant_text_fr_state_change(curr_interaction_obj, sentences, is_state_change=True)
            if source_text is None:
                source_text = ''
        else:
            source_text = ''
        curr_interaction_obj.text_sentence = source_text
        state_change_interactions_objs_list.append(curr_interaction_obj)
    extend_prior_model.interaction_obj_to_darpa_json(state_change_interactions_objs_list, file_path=file_path, is_append=is_append)


def write_protein_state_changes(X, protein_names, state_names, file_path, X_lkl=None, is_append=False, amr_file=None, model_interactions=None, passage_name=None, is_csv=False, is_model=False, model_protein_names=None, sentences=None):
    raise DeprecationWarning
    #X is an array of protein state changes
    n_x = X.shape[0]
    if is_append:
        f = open(cap.absolute_path+file_path, 'a')
    else:
        f = open(cap.absolute_path+file_path, 'w')
    if is_csv:
        if not is_model:
            field_names = ['Passage','Relation','English-Like Description', 'Model Representation','Model Link','Source Text']
            # f.write('Passage,Relation,English-Like Description,Model Representation,Model Link,Source Text\n')
        else:
            field_names = ['English-Like Description', 'Model Representation']
            # f.write('English-Like Description,Model Representation\n')
        csv_writer = csv.DictWriter(f, fieldnames=field_names, dialect='excel', lineterminator='\n', delimiter=',')
        if not is_append:
            csv_writer.writeheader()
    else:
        if amr_file is not None and not is_append:
            f.write('\nAMR ID: ' + get_amr_id(amr_file) + '\n')
        if not is_append:
            f.write('\nNo of proteins in AMRs including None: ' + str(protein_names.size-1))
            f.write('\nNo of state bits in AMRs: ' + str(state_names.size))
        if X_lkl is None:
            f.write('\n\n# ' + str(n_x) + ' no. of protein state change interactions follow\n\n')
        else:
            f.write('\n\n# ' + str(n_x) + ' no. of protein state change interactions with inference likelihood in decreasing order follow\n\n')
    for curr_idx in np.arange(n_x):
        x = X[curr_idx]
        if X_lkl is not None:
            lkl = X_lkl[curr_idx]
        else:
            lkl = None
        curr_interaction = get_state_change_interaction_obj(protein_names, state_names, x, lkl)
        # print 'str(curr_interaction):', str(curr_interaction)
        if (curr_interaction.catalyst_str is not None and curr_interaction.catalyst_str) or curr_interaction.is_mechanistic_information() or allow_state_change_without_catalyst_nonmechanistic:
            if not curr_interaction.contains_nonentity():
                out_eng = curr_interaction.english()
                out_str = str(curr_interaction)
                if model_interactions is not None and not is_model:
                    if curr_interaction.catalyst_str is None or not curr_interaction.catalyst_str:
                        if curr_interaction.is_mechanistic_information():
                            interaction_relation = 'New Mechanistic Information'
                        elif allow_state_change_without_catalyst_nonmechanistic:
                            interaction_relation = 'New Relational Information'
                        else:
                            raise AssertionError
                    else:
                        interaction_relation = 'New Relational Information'
                    if state_change in model_interactions and model_interactions[state_change]:
                        if curr_interaction in model_interactions[state_change]:
                            interaction_relation = 'Corroborating Information'
                        else:
                            curr_interaction_conflict = copy.deepcopy(curr_interaction)
                            curr_interaction_conflict.is_positive_catalyst = not curr_interaction_conflict.is_positive_catalyst
                            if curr_interaction_conflict in model_interactions[state_change]:
                                interaction_relation = 'Conflicting Information'
                else:
                    interaction_relation = ''
                if passage_name is None:
                    passage_name = ''
                if model_protein_names is not None and model_protein_names:
                    link_list = ''
                    if curr_interaction.catalyst_str in model_protein_names:
                        link_list += curr_interaction.catalyst_str
                        link_list += ','
                    if curr_interaction.protein_str in model_protein_names:
                        link_list += curr_interaction.protein_str
                        link_list += ','
                    link_list = link_list.rstrip(',')
                    # if link_list == '[]':
                    #     link_list = ''
                else:
                    link_list = ''
                if sentences is not None:
                    source_text = get_relevant_text_fr_state_change(curr_interaction, sentences, is_state_change=True)
                    if source_text is None:
                        source_text = ''
                else:
                    source_text = ''
                if not is_csv:
                    f.write(out_eng+'\n')
                    f.write(out_str+'\n')
                    if not is_model:
                        if interaction_relation:
                            f.write('Relationship to Model: ' + interaction_relation+'\n')
                        if passage_name:
                            f.write(passage_name+'\n')
                        if link_list:
                            f.write('Link: ' + link_list + '\n')
                        if source_text:
                            f.write('Source Text: ' + source_text + '\n')
                    f.write('\n\n\n')
                else:
                    if not is_model:
                        csv_writer.writerow({'Passage': passage_name, 'Relation':interaction_relation, 'English-Like Description':out_eng, 'Model Representation':out_str, 'Model Link':link_list, 'Source Text':source_text})
                        # f.write(passage_name + ',' + interaction_relation + ',' + out_eng + ',' + out_str + ',' + link_list + ',' + source_text)
                    else:
                        csv_writer.writerow({'English-Like Description':out_eng, 'Model Representation':out_str})
                        # f.write(out_eng + ',' + out_str)
                    # f.write('\n')
    f.close()


def get_relevant_text_fr_state_change(interaction, sentences, is_state_change):
    passage = ''
    for sentence in sentences:
        if sentence is not None and sentence:
            is_relevant = True
            if interaction.catalyst_str is not None and interaction.catalyst_str not in sentence.upper():
                is_relevant = False
            if is_state_change:
                if interaction.protein_str is not None and interaction.protein_str not in sentence.upper():
                    is_relevant = False
            else:
                if interaction.protein_1_str is not None and interaction.protein_1_str not in sentence.upper():
                    is_relevant = False
                if interaction.protein_2_str is not None and interaction.protein_2_str not in sentence.upper():
                    is_relevant = False
            if is_relevant:
                passage += no_quotes_regexp.sub('', sentence)
    if passage:
        return passage


def write_protein_state_changes_sif(X, protein_names, state_names, file_path, X_lkl=None, is_append=False, amr_file=None):
    #X is an array of protein state changes
    n_x = X.shape[0]
    if is_append:
        f = open(cap.absolute_path+file_path, 'a')
    else:
        f = open(cap.absolute_path+file_path, 'w')
    for curr_idx in np.arange(n_x):
        x = X[curr_idx]
        if X_lkl is not None:
            lkl = X_lkl[curr_idx]
        else:
            lkl = None
        out_str = interaction_to_str_sif(protein_names, state_names, x, lkl)
        f.write(out_str+'\n')
    f.close()


def write_json_index_cards_protein_complex_formations(X, protein_names, state_names, file_path, X_lkl=None, is_append=False, amr_file=None, model_interactions=None, passage_name=None, is_model=False, model_protein_names=None, sentences=None):
    #X is an array of protein state changes
    n_x = X.shape[0]
    k = state_names.size
    complex_form_objs_list = []
    for curr_idx in np.arange(n_x):
        x = X[curr_idx]
        if X_lkl is not None:
            lkl = X_lkl[curr_idx]
        else:
            lkl = None
        curr_complex_form_obj = get_complex_form_obj(protein_names, state_names, x, lkl)
        if model_interactions is not None and not is_model:
            interaction_relation = 'Extension'
            if complex in model_interactions and model_interactions[complex]:
                if curr_complex_form_obj in model_interactions[complex]:
                    interaction_relation = 'Corroborating'
                else:
                    curr_complex_form_conflict = copy.deepcopy(curr_complex_form_obj)
                    curr_complex_form_conflict.is_positive_catalyst = not curr_complex_form_conflict.is_positive_catalyst
                    if curr_complex_form_conflict in model_interactions[complex]:
                        interaction_relation = 'Conflicting'
                    else:
                        curr_complex_form_no_catalyst = copy.deepcopy(curr_complex_form_obj)
                        curr_complex_form_no_catalyst.catalyst_str = None
                        curr_complex_form_no_catalyst.catalyst_state_str_list = []
                        if curr_complex_form_no_catalyst in model_interactions[complex]:
                            interaction_relation = 'Specialization'
        else:
            interaction_relation = ''
        curr_complex_form_obj.model_relation = interaction_relation
        if model_protein_names is not None and model_protein_names:
            link_list = ''
            if curr_complex_form_obj.catalyst_str in model_protein_names:
                link_list += curr_complex_form_obj.catalyst_str
                link_list += ','
            if curr_complex_form_obj.protein_1_str in model_protein_names:
                link_list += curr_complex_form_obj.protein_1_str
                link_list += ','
            if curr_complex_form_obj.protein_2_str in model_protein_names:
                link_list += curr_complex_form_obj.protein_2_str
                link_list += ','
            if curr_complex_form_obj.complex_str in model_protein_names:
                link_list += curr_complex_form_obj.complex_str
                link_list += ','
            link_list = link_list.rstrip(',')
        else:
            link_list = ''
        curr_complex_form_obj.protein_comma_list_link_to_model = link_list
        if sentences is not None:
            source_text = get_relevant_text_fr_state_change(curr_complex_form_obj, sentences, is_state_change=False)
            if source_text is None:
                source_text = ''
        else:
            source_text = ''
        curr_complex_form_obj.text_sentence = source_text
        complex_form_objs_list.append(curr_complex_form_obj)
    return complex_form_objs_list


def write_protein_complex_formations(X, protein_names, state_names, file_path, X_lkl=None, is_append=False, amr_file=None, model_interactions=None, passage_name=None, is_csv=False, is_model=False, model_protein_names=None, sentences=None):
    #X is an array of protein state changes
    n_x = X.shape[0]
    k = state_names.size
    if is_append:
        f = open(cap.absolute_path+file_path, 'a')
    else:
        f = open(cap.absolute_path+file_path, 'w')
    if is_csv:
        if not is_model:
            field_names = ['Passage','Relation','English-Like Description','Model Representation','Model Link','Source Text']
            # f.write('Passage,Relation,English-Like Description,Model Representation,Model Link,Source Text\n')
        else:
            field_names = ['English-Like Description', 'Model Representation']
            # f.write('English-Like Description,Model Representation\n')
        csv_writer = csv.DictWriter(f, fieldnames=field_names, dialect='excel', lineterminator='\n', delimiter=',')
        if not is_append:
            csv_writer.writeheader()
    else:
        if amr_file is not None and not is_append:
            f.write('\nAMR ID: ' + get_amr_id(amr_file) + '\n')
        if not is_append:
            f.write('\nNo of proteins in AMRs including None: ' + str(protein_names.size-1))
            f.write('\nNo of state bits in AMRs: ' + str(state_names.size))
        if X_lkl is None:
            f.write('\n\n# ' + str(n_x) + ' no. of protein complex formation interactions follow\n')
        else:
            f.write('\n\n# ' + str(n_x) + ' no. of protein complex formation interactions with inference likelihood in decreasing order follow\n')
    for curr_idx in np.arange(n_x):
        x = X[curr_idx]
        if X_lkl is not None:
            lkl = X_lkl[curr_idx]
        else:
            lkl = None
        curr_complex_form = get_complex_form_obj(protein_names, state_names, x, lkl)
        if not curr_complex_form.contains_nonentity():
            out_eng = curr_complex_form.english()
            out_str = str(curr_complex_form)
            if model_interactions is not None and not is_model:
                interaction_relation = 'New Relational Information'
                if state_change in model_interactions and model_interactions[state_change]:
                    if curr_complex_form in model_interactions[complex]:
                        interaction_relation = 'Corroborating Information'
                    else:
                        curr_complex_form_conflict = copy.deepcopy(curr_complex_form)
                        curr_complex_form_conflict.is_positive_catalyst = not curr_complex_form_conflict.is_positive_catalyst
                        if curr_complex_form_conflict in model_interactions[complex]:
                            interaction_relation = 'Conflicting Information'
                        else:
                            curr_complex_form_no_catalyst = copy.deepcopy(curr_complex_form)
                            curr_complex_form_no_catalyst.catalyst_str = None
                            curr_complex_form_no_catalyst.catalyst_state_str_list = []
                            if curr_complex_form_no_catalyst in model_interactions[complex]:
                                interaction_relation = 'Specialization'
            else:
                interaction_relation = ''
            if passage_name is None:
                passage_name = ''
            if model_protein_names is not None and model_protein_names:
                link_list = ''
                if curr_complex_form.catalyst_str in model_protein_names:
                    link_list += curr_complex_form.catalyst_str
                    link_list += ','
                if curr_complex_form.protein_1_str in model_protein_names:
                    link_list += curr_complex_form.protein_1_str
                    link_list += ','
                if curr_complex_form.protein_2_str in model_protein_names:
                    link_list += curr_complex_form.protein_2_str
                    link_list += ','
                if curr_complex_form.complex_str in model_protein_names:
                    link_list += curr_complex_form.complex_str
                    link_list += ','
                link_list = link_list.rstrip(',')
                # link_list += ']'
                # if link_list == '[]':
                #     link_list = ''
            else:
                link_list = ''
            if sentences is not None:
                source_text = get_relevant_text_fr_state_change(curr_complex_form, sentences, is_state_change=False)
                if source_text is None:
                    source_text = ''
            else:
                source_text = ''
            if is_csv:
                if not is_model:
                    csv_writer.writerow({'Passage': passage_name, 'Relation':interaction_relation, 'English-Like Description':out_eng, 'Model Representation':out_str, 'Model Link':link_list, 'Source Text':source_text})
                    # f.write(passage_name + ',' + interaction_relation + ',' + out_eng + ',' + out_str + ',' + link_list + ',' + source_text)
                else:
                    csv_writer.writerow({'English-Like Description':out_eng, 'Model Representation':out_str})
                    # f.write(out_eng + ',' + out_str)
                # f.write('\n')
            else:
                f.write(out_eng+'\n')
                f.write(out_str+'\n')
                if not is_model:
                    if interaction_relation:
                        f.write('Relationship to Model: ' + interaction_relation)
                        f.write('\n')
                    if passage_name:
                        f.write(passage_name+'\n')
                    if link_list:
                        f.write('Link: ' + link_list + '\n')
                    if source_text:
                        f.write('Source Text: ' + source_text + '\n')
                f.write('\n\n\n')
    if is_csv:
        csv_writer = None
    f.close()


def write_protein_complex_formations_sif(X, protein_names, state_names, file_path, X_lkl=None, is_append=False, amr_file=None):
    #X is an array of protein state changes
    n_x = X.shape[0]
    k = state_names.size
    if is_append:
        f = open(cap.absolute_path+file_path, 'a')
    else:
        f = open(cap.absolute_path+file_path, 'w')
    for curr_idx in np.arange(n_x):
        x = X[curr_idx]
        if X_lkl is not None:
            lkl = X_lkl[curr_idx]
        else:
            lkl = None
        out_str = interaction_to_str_sif(protein_names, state_names, x, lkl)
        f.write(out_str+'\n')
    f.close()


#noise model is general
def generate_noise_model(n, is_normalize=True):
    print 'generating noise model ...'
    print 'No. of proteins n (input parameter) is: ', n
    # herein, we assume noise only at protein level, not at state level. It is also assumed that system is able recognize type of interaction (either state change (which itself includes many subtypes) or complex formation)
    # number of proteins
    #in addition, we have two additional proteins: NULL, GARBAGE
    # index 0 refers to NULL protein (i.e. protein doesn't exist for a position in the protein interaction)
    # index -1 refers to GARBAGE protein, a protein which can not be identified as a valid protein
    phi_ijkl = 0.1*np.ones([n+1, n+2]) #_ijkl means that the phi noise model is relevant for any protein position i, j, k, l
    # in state transformation, i is position for protein which catalyzes/activates the state change
    # j is position of protein whose state is changed
    # in complex formation, i refers to catalyst
    # j refers to a protein forming complex
    # k refers to a protein forming complex
    # l refers to the new complex
    # except for null and garbage proteins, there is high probability for observing same protein entity
    # for now, setting diagonal to high value,
    # for null, and garbage, the values are updated later on
    phi_ijkl[np.diag_indices(n+1)] = 0.9
    # print 'phi_ijkl: ', phi_ijkl
    # phi_ijkl[0, :] = 0.15 #observing a not null protein with ground truth null protein
    # phi_ijkl[0, -1] = 0.15 #observing garbage protein with ground truth null protein
    # phi_ijkl[0, 0] = 0.7 #observing null from ground truth null is less probable
    # phi_ijkl[1:, -1] = 0.2 #observing a garbage value from non null protein is also less probable
    #normalizing protein observation probabilities
    #assuming that no probability vector for a protein (no. of ground proteins is n-1, while excluding GARBAGE protein) is a zero vector
    if is_normalize:
        phi_ijkl /= np.tile(phi_ijkl.sum(1).reshape([n+1, 1]), (1, n+2))
    #printing the noise model
    # print 'Noise model at protein level (phi_ijkl) is: '
    # print 'This noise model (phi_ijkl) represents probability of observing a protein instead of ground truth protein at any position i, j, k, l in protein interactions'
    # print 'Number of ground truth proteins, including NULL, is: ', n+1
    # print 'Number of observed proteins, including NULL and GARBAGE, is: ', n+2
    # print 'index 0 is for protein NULL'
    # print 'index -1 is for protein GARBAGE (unidentified protein)'
    # print phi_ijkl
    #now, we need to define noise at level of proteins
    #noise for proteins at positions i and j
    #dimension 0 correspond to protein i ground truth
    #dimension 1 correspond to protein j or k ground truth
    #dimension 2 correspond to protein i observed
    #dimension 3 correspond to protein j or k observed
    phi_i_jk = 0.1*np.ones([n+1, n+1, n+2, n+2])
    phi_i_jk[:, :, -1, :] = 0.01 #probability of observing garbage through exchange noise model is zero
    phi_i_jk[:, :, :, -1] = 0.01 #same here
    #probability of swapping between proteins i, j and i, k is high
    #todo: try to avoid the loops
    #iterating through proteins for position i
    for i_idx in np.arange(0, n+1):
        #iterating through proteins for position j/k
        for jk_idx in np.arange(1, n+1): #same here
            if i_idx == 0:
                phi_i_jk[i_idx, jk_idx, jk_idx, jk_idx] = 0.3 #probability of observing j/k as i (i is catalyst) when ground truth i is null
            else:
                phi_i_jk[i_idx, jk_idx, i_idx, jk_idx] = 0.8 #probability of observing ground truth itself
                phi_i_jk[i_idx, jk_idx, jk_idx, i_idx] = 0.2 #probability of swapping
    #normalizing the vectors
    if is_normalize:
        phi_i_jk /= np.tile(phi_i_jk.sum((2, 3)).reshape([n+1, n+1, 1, 1]), (1, 1, n+2, n+2))
    #printing the noise model
    # print 'Noise model at protein interaction level (phi_i_jk) is: '
    # print 'This noise model (phi_i_jk) represents probability of observing a set of proteins at positions {i,j}/{i,k} instead of the ground truth protein pairs'
    # print 'Number of ground truth proteins, including NULL, is: ', n+1
    # print 'Number of observed proteins, including NULL and GARBAGE, is: ', n+2
    # print 'index 0 is for protein NULL'
    # print 'index -1 is for protein GARBAGE'
    # print phi_i_jk
    return phi_ijkl, phi_i_jk


#noise model is general
def generate_noise_state_model(m, is_normalize=True):
    print 'generating noise model for state ...'
    print 'No. of states m (input parameter) is: ', m
    phi_s = 0.1*np.ones([m, m])
    phi_s[np.diag_indices(m)] = 0.8
    if is_normalize:
        phi_s /= np.tile(phi_s.sum(1).reshape([m, 1]), (1, m))
    return phi_s


def init_states_set():
    k = 2
    state_names = np.empty(shape=k, dtype=np.object)
    state_names[0] = 'phosphorylated' #bit zero in a boolean state vector
    state_names[1] = 'complex' #bit one in a boolean state vector
    print 'State flags are: '
    print state_names
    return state_names


def get_state_vector_from(states_int, k):
    #k is no. of state bits
    # print 'No. of state bits is: ', k
    #states_int is a vector of ints
    #state of a protein is a vector where each bit represents the corresponding state flag. For instance, phosphorylation represents is one state flag with value 1 representing phosphorylation
    if not type(states_int) == np.ndarray:
        if isinstance(states_int, list):
            states_int = np.array(states_int)
        elif isinstance(states_int, (int, long)):
            states_int = np.array([states_int])
    n = states_int.size
    states_bool = np.zeros(shape=(n, k)).astype(np.bool)
    for i in range(n):
        curr_bits_str = np.binary_repr(states_int[i])
        m = len(curr_bits_str)
        for j in range(1, m+1):
            states_bool[i, -j] = bool(int(curr_bits_str[-j]))
    if states_bool.shape[0] == 1:
        states_bool = states_bool[0]
    # print 'states_bool is: ', states_bool
    return states_bool


# def get_state_vector_from_backup(states_int, k):
#     #k is no. of state bits
#     # print 'No. of state bits is: ', k
#     if k > 8:
#         raise NotImplementedError
#     #todo: in the future, for higher no. of state bits (>8), we can use numpy.binary_repr
#     #states_int is a vector of ints
#     #state of a protein is a vector where each bit represents the corresponding state flag. For instance, phosphorylation represents is one state flag with value 1 representing phosphorylation
#     try:
#         n = states_int.size
#     except AttributeError:
#         states_int = np.array(states_int)
#         n = states_int.size
#     states_bool = np.unpackbits(states_int.reshape([n, 1]).astype(np.uint8), axis=1).astype(np.bool)
#     states_bool = states_bool[:, -k:]
#     if states_bool.shape[0] == 1:
#         states_bool = states_bool[0]
#     # print 'states_bool is: ', states_bool
#     return states_bool


def get_state_string_vector_from(state_bool, state_names):
    #state_bool is a single state vector
    k = state_bool.astype(np.int).sum()
    if k == 0:
        return np.array([])
    else:
        state_string_flags = np.empty(shape=k, dtype=np.object)
        count = 0
        for curr_idx in np.arange(state_bool.size):
            if state_bool[curr_idx] == True:
                state_string_flags[count] = state_names[curr_idx].strip('\'')
                count += 1
        return state_string_flags


def get_state_int_from(states_bool):
    #states_bool is a 2-D array with first dimension representing each state vector and second dimension representing state bit for a given state vector
    # print 'states_bool:', states_bool
    if debug:
        print 'states_bool', states_bool
    (n, k) = states_bool.shape
    if debug:
        print '(n, k)', (n, k)
    states_int = np.zeros(shape=(n, 1), dtype=long)
    for i in range(n):
        # states_int[i] = 0
        for j in range(k):
            if states_bool[i,-(j+1)]:
                states_int[i] = states_int[i] + 2**j
    return states_int


# def get_state_int_from_backup(states_bool):
#     #todo: in the future, for higher no. of state bits (>8), we can use numpy.binary_repr related api
#     #states_bool is a 2-D array with first dimension representing each state vector and second dimension representing state bit for a given state vector
#     # print 'states_bool:', states_bool
#     return np.packbits(states_bool.astype(np.uint8))


def init_proteins_set():
    n = 19
    protein_names = np.empty(shape=(n+2), dtype=np.object)
    protein_names[0] = 'NULL'
    protein_names[-1] = 'GARBAGE'
    protein_names[1] = 'TP53'
    protein_names[2] = 'ATM'
    protein_names[3] = 'RAD50'
    protein_names[4] = 'MRE11'
    protein_names[5] = 'RAD50:MRE11'
    protein_names[6] = 'NBN'
    protein_names[7] = 'MRE11:RAD50:NBS1'
    protein_names[8] = 'gamma-H2AX:NBS1'
    protein_names[9] = 'MRN'
    protein_names[10] = 'MDC1/NFBD1:gamma-H2AX'
    protein_names[11] = 'ATMassociatedwithDNAdouble-strandbreakends'
    protein_names[12] = 'gammaH2AX-coatedDNAdouble-strandbreakends'
    protein_names[13] = 'H2AFX'
    protein_names[14] = 'BRCA1'
    protein_names[15] = 'MDC1'
    protein_names[16] = '53BP1:H2AX'
    protein_names[17] = 'BRCA1:53BP1'
    protein_names[18] = 'gammaH2AX:MDC1/NFBD1'
    protein_names[19] = 'TP53BP1'
    print 'Protein names are: '
    print protein_names
    return protein_names


def get_protein_and_state_names_from_interactions_list(interactions):
    def add_protein_name(name):
        if name is not None:
            if not protein_names.has_key(name):
                protein_names[name] = name

    def add_state_name(name_list):
        if (name_list is not None) and name_list:
            for name in name_list:
                if name is not None:
                    state_names[name] = name

    protein_names = {}
    state_names = {}
    #state change interactions
    if interactions.has_key(state_change) and (interactions[state_change] is not None):
        for state_change_interaction in interactions[state_change]:
            add_protein_name(state_change_interaction.catalyst_str)
            add_protein_name(state_change_interaction.protein_str)
            add_state_name(state_change_interaction.catalyst_state_str_list)
            add_state_name(state_change_interaction.protein_state_str_list)
            add_state_name(state_change_interaction.result_state_str_list)
            if not state_change_interaction.is_positive_catalyst:
                add_state_name([inhibit])
    if interactions.has_key(complex) and (interactions[complex] is not None):
        for complex_interaction in interactions[complex]:
            add_protein_name(complex_interaction.catalyst_str)
            add_protein_name(complex_interaction.protein_1_str)
            add_protein_name(complex_interaction.protein_2_str)
            add_protein_name(complex_interaction.complex_str)
            add_state_name(complex_interaction.protein_1_state_str_list)
            add_state_name(complex_interaction.protein_2_state_str_list)
            add_state_name(complex_interaction.catalyst_state_str_list)
            add_state_name(complex_interaction.complex_state_str_list)
            if not complex_interaction.is_positive_catalyst:
                add_state_name([inhibit])
    proteins = np.empty(dtype=np.object, shape=len(protein_names)+2)
    proteins[0] = NULL
    proteins[-1] = GARBAGE
    proteins[1:-1] = np.array(protein_names.values()).flatten()
    states = np.array(state_names.values()).flatten()
    return proteins, states


def init_gen_proteins_state_change_model_sparse(n, k, model_interactions=None):
    #n- no. of proteins
    #k number of state bits including None
    m = 2**k #no. of possible states in state bit space
    # first dimension of psi represents protein for position i (i.e. catalyst)
    # and second dimension represents state of the protein at position i
    # and third dimension represents protein at position j (i.e. protein whose state is changed)
    # and fourth dimension represents previous state (state prior to state transition) for protein at position j
    # and fifth dimension represents new state (state post state transition) for protein at position j
    #psi_i_j denotes that the model involves proteins at positions i and j
    #index 0 correspond to None for proteins
    psi_shape = (n+1, m, n+1, m, m, 2)
    psi_i_j = sa.sparray(shape=psi_shape, default=interaction_prior_value, dtype=np.float)
    if model_interactions is not None:
        psi_i_j[model_interactions] = 0.9
    #there should be a state transition
    # for curr_state_idx in range(m):
    #     psi_i_j[:, :, :, curr_state_idx, curr_state_idx] = 0
    #also, we are assuming that catalyst and protein can not be same
    # for curr_protein_idx in range(n+1):
    #     psi_i_j[curr_protein_idx, :, curr_protein_idx, :, :] = 0
    # #probability for an interaction without a catalyst is low
    # psi_i_j[0] *= 0.3
    # #protein can not be None at all
    # psi_i_j[:, :, 0] = 0
    return psi_i_j


def init_gen_proteins_state_change_model(n, k):
    #n- no. of proteins
    #k number of state bits including None
    m = 2**k #no. of possible states in state bit space
    # first dimension of psi represents protein for position i (i.e. catalyst)
    # and second dimension represents state of the protein at position i
    # and third dimension represents protein at position j (i.e. protein whose state is changed)
    # and fourth dimension represents previous state (state prior to state transition) for protein at position j
    # and fifth dimension represents new state (state post state transition) for protein at position j
    #psi_i_j denotes that the model involves proteins at positions i and j
    #index 0 correspond to None for proteins
    psi_i_j = interaction_prior_value*np.ones([n+1, m, n+1, m, m, 2], dtype=np.float) #bernoulli trial
    # #since it is a protein state change interaction, interactions with no state change in the matrix will have very low probability
    # psi_shape = psi_i_j.shape
    # for curr_state_idx in range(m):
    #     curr_state_idx_arr = np.array([curr_state_idx])
    #     mesh_idx = np.meshgrid(np.arange(psi_shape[0]), np.arange(psi_shape[1]), np.arange(psi_shape[2]), curr_state_idx_arr, curr_state_idx_arr, indexing='ij')
    #     psi_i_j[mesh_idx] = 0
    #     if psi_i_j[:, :, :, curr_state_idx, curr_state_idx].sum() != 0:
    #         raise AssertionError
    # del curr_state_idx_arr
    # #also, we are assuming that catalyst and protein can not be same
    # for curr_protein_idx in range(n+1):
    #     curr_protein_idx_arr = np.array([curr_protein_idx])
    #     mesh_idx = np.meshgrid(curr_protein_idx_arr, np.arange(psi_shape[1]), curr_protein_idx_arr, np.arange(psi_shape[3]), np.arange(psi_shape[4]), indexing='ij')
    #     psi_i_j[mesh_idx] = 0
    #     if psi_i_j[curr_protein_idx, :, curr_protein_idx, :, :].sum() != 0:
    #         raise AssertionError
    # del curr_protein_idx_arr
    # #probability for an interaction without a catalyst is low
    # psi_i_j[0] *= 0.3
    # #protein can not be None at all
    # psi_i_j[:, :, 0] = 0
    return psi_i_j


def init_gen_proteins_complex_model_sparse(n, k, model_interactions=None):
    #n- no. of proteins
    #k number of state bits including None
    m = 2**k #no. of possible states in state bit space
    # first dimension of psi represents protein for position i (i.e. catalyst)
    # and second dimension represents state of the protein at position i
    # and third dimension represents protein at position j (i.e. protein1 forming complex)
    # and fourth dimension represents  state for protein at position j
    # and fifth dimension represents protein at position k (i.e. protein2 forming complex)
    # and sixth dimension represents  state for protein at position k
    # and seventh dimension represents protein at position l (i.e. protein complex)
    # and eighth dimension represents  state for protein-complex at position l
    #psi_i_jk_l denotes that the model involves proteins at positions i, j, k, l
    #index 0 correspond to None for proteins
    psi_shape = (n+1, m, n+1, m, n+1, m, n+1, m, 2)
    psi_i_jk_l = sa.sparray(shape=psi_shape, default=interaction_prior_value, dtype=np.float)
    if model_interactions is not None:
        psi_i_jk_l[model_interactions] = 0.9
    # #we are assuming that catalyst i and protein at {j, k, l} can not be same
    # #also, protein 1 and protein 2 can not same
    # #also, protein 1 and complex can not be same
    # #also, protein 2 and complex can not be same
    # psi_shape = psi_i_jk_l.shape
    # for curr_protein_idx in range(n+1):
    #     curr_protein_idx_arr = np.array([curr_protein_idx])
    #     mesh_idx = (curr_protein_idx_arr, np.arange(psi_shape[1]), curr_protein_idx_arr, np.arange(psi_shape[3]), np.arange(psi_shape[4]))
    #     psi_i_jk_l[mesh_idx] = 0
    #     mesh_idx = (curr_protein_idx_arr, np.arange(psi_shape[1]), np.arange(psi_shape[2]), np.arange(psi_shape[3]), curr_protein_idx_arr)
    #     psi_i_jk_l[mesh_idx] = 0
    #     mesh_idx = (curr_protein_idx_arr, np.arange(psi_shape[1]), np.arange(psi_shape[2]), np.arange(psi_shape[3]), np.arange(psi_shape[4]), np.arange(psi_shape[5]), curr_protein_idx_arr)
    #     psi_i_jk_l[mesh_idx] = 0
    #     # mesh_idx = (np.arange(psi_shape[0]), np.arange(psi_shape[1]), curr_protein_idx_arr, np.arange(psi_shape[3]), curr_protein_idx_arr)
    #     # psi_i_jk_l[mesh_idx] = 0
    #     # mesh_idx = (np.arange(psi_shape[0]), np.arange(psi_shape[1]), curr_protein_idx_arr, np.arange(psi_shape[3]), np.arange(psi_shape[4]), np.arange(psi_shape[5]), curr_protein_idx_arr)
    #     # psi_i_jk_l[mesh_idx] = 0
    #     # mesh_idx = (np.arange(psi_shape[0]), np.arange(psi_shape[1]), np.arange(psi_shape[2]), np.arange(psi_shape[3]), curr_protein_idx_arr, np.arange(psi_shape[5]), curr_protein_idx_arr)
    #     # psi_i_jk_l[mesh_idx] = 0
    # del curr_protein_idx_arr
    # # #probability for an interaction without a catalyst is high
    # # psi_i_jk_l[0] /= 0.8
    # # #protein1, protein2, complex can not be None at all
    # # psi_i_jk_l[:, :, 0] = 0
    # # psi_i_jk_l[:, :, :, :, 0] = 0
    # # psi_i_jk_l[:, :, :, :, :, :, 0] = 0
    return psi_i_jk_l


def init_gen_proteins_complex_model(n, k):
    #n- no. of proteins
    #k number of state bits including None
    m = 2**k #no. of possible states in state bit space
    # first dimension of psi represents protein for position i (i.e. catalyst)
    # and second dimension represents state of the protein at position i
    # and third dimension represents protein at position j (i.e. protein1 forming complex)
    # and fourth dimension represents  state for protein at position j
    # and fifth dimension represents protein at position k (i.e. protein2 forming complex)
    # and sixth dimension represents  state for protein at position k
    # and seventh dimension represents protein at position l (i.e. protein complex)
    # and eighth dimension represents  state for protein-complex at position l
    #psi_i_jk_l denotes that the model involves proteins at positions i, j, k, l
    #index 0 correspond to None for proteins
    psi_i_jk_l = interaction_prior_value*np.ones([n+1, m, n+1, m, n+1, m, n+1, m, 2], dtype=np.float)
    #we are assuming that catalyst i and protein at {j, k, l} can not be same
    #also, protein 1 and protein 2 can not same
    #also, protein 1 and complex can not be same
    #also, protein 2 and complex can not be same
    psi_shape = psi_i_jk_l.shape
    for curr_protein_idx in range(n+1):
        curr_protein_idx_arr = np.array([curr_protein_idx])
        #
        mesh_idx = np.meshgrid(curr_protein_idx_arr, np.arange(psi_shape[1]), curr_protein_idx_arr, np.arange(psi_shape[3]), np.arange(psi_shape[4]), indexing='ij')
        psi_i_jk_l[mesh_idx] = 0
        if psi_i_jk_l[curr_protein_idx, :, curr_protein_idx].sum() != 0:
            raise AssertionError
        #
        mesh_idx = np.meshgrid(curr_protein_idx_arr, np.arange(psi_shape[1]), np.arange(psi_shape[2]), np.arange(psi_shape[3]), curr_protein_idx_arr, indexing='ij')
        psi_i_jk_l[mesh_idx] = 0
        if psi_i_jk_l[curr_protein_idx, :, :, :, curr_protein_idx].sum() != 0:
            raise AssertionError
        #
        mesh_idx = np.meshgrid(curr_protein_idx_arr, np.arange(psi_shape[1]), np.arange(psi_shape[2]), np.arange(psi_shape[3]), np.arange(psi_shape[4]), np.arange(psi_shape[5]), curr_protein_idx_arr, indexing='ij')
        psi_i_jk_l[mesh_idx] = 0
        if psi_i_jk_l[curr_protein_idx, :, :, :, :, :, curr_protein_idx].sum() != 0:
            raise AssertionError
        #
        mesh_idx = np.meshgrid(np.arange(psi_shape[0]), np.arange(psi_shape[1]), curr_protein_idx_arr, np.arange(psi_shape[3]), curr_protein_idx_arr, indexing='ij')
        psi_i_jk_l[mesh_idx] = 0
        if psi_i_jk_l[:, :, curr_protein_idx, :, curr_protein_idx].sum() != 0:
            raise AssertionError
        #
        mesh_idx = np.meshgrid(np.arange(psi_shape[0]), np.arange(psi_shape[1]), curr_protein_idx_arr, np.arange(psi_shape[3]), np.arange(psi_shape[4]), np.arange(psi_shape[5]), curr_protein_idx_arr, indexing='ij')
        psi_i_jk_l[mesh_idx] = 0
        if psi_i_jk_l[:, :, curr_protein_idx, :, :, :, curr_protein_idx].sum() != 0:
            raise AssertionError
        #
        mesh_idx = np.meshgrid(np.arange(psi_shape[0]), np.arange(psi_shape[1]), np.arange(psi_shape[2]), np.arange(psi_shape[3]), curr_protein_idx_arr, np.arange(psi_shape[5]), curr_protein_idx_arr, indexing='ij')
        psi_i_jk_l[mesh_idx] = 0
        if psi_i_jk_l[:, :, :, :, curr_protein_idx, :, curr_protein_idx].sum() != 0:
            raise AssertionError
    del curr_protein_idx_arr
    # #probability for an interaction without a catalyst is high
    # psi_i_jk_l[0] /= 0.8
    #protein1, protein2, complex can not be None at all
    psi_i_jk_l[:, :, 0] = 0
    psi_i_jk_l[:, :, :, :, 0] = 0
    psi_i_jk_l[:, :, :, :, :, :, 0] = 0
    # print 'Protein state change model (psi_i_jk_l) is: '
    # print psi_i_jk_l
    return psi_i_jk_l


#protein interaction model is not general and specific to proteins. Though, in the future, the proteins and states can be represented as features and correspondingly probability of an interaction can be represented as categorical output represented in a non-linear model such as neural networks etc
def init_proteins_state_change_model(X):
    #X is ground truth state changes
    k = 2 #number of state bits
    m = 2**k #no. of possible states in state bit space
    n = 19
    # first dimension of psi represents protein for position i (i.e. catalyst)
    # and second dimension represents state of the protein at position i
    # and third dimension represents protein at position j (i.e. protein whose state is changed)
    # and fourth dimension represents previous state (state prior to state transition) for protein at position j
    # and fifth dimension represents new state (state post state transition) for protein at position j
    #psi_i_j denotes that the model involves proteins at positions i and j
    psi_i_j = 0.1*np.ones([n+1, m, n+1, m, m])
    n_x = X.shape[0]
    for curr_idx in np.arange(n_x):
        x = X[curr_idx]
        psi_i_j[tuple(x)] = 0.8
    #the explicit assignment below is not required
    # # Phophorylated protein complex ATM phosphorylates protein TP53
    # psi_i_j[2, 3, 1, 0, 2] = 0.8
    # #Protein BRCA1 phosphorylates protein NBN
    # psi_i_j[14, 0, 6, 0, 2] = 0.8
    # #Protein MDC1 phosphorylates protein BRCA1
    # psi_i_j[15, 0, 14, 0, 2] = 0.8
    # #Protein MDC1 is phosphorylated
    # psi_i_j[0, 0, 15, 0, 2] = 0.8
    # #Protein H2AFX is phosphorylated.
    # psi_i_j[0, 0, 13, 0, 2] = 0.8
    #normalizing
    psi_i_j /= psi_i_j.sum()
    print 'Protein state change model (psi_i_j) is: '
    print psi_i_j
    return psi_i_j


def init_ground_truth_protein_state_changes():
    n = 5
    X = np.zeros([n, 5], np.int)
    # Phophorylated protein complex ATM phosphorylates protein TP53
    X[0] = np.array([2, 3, 1, 0, 2])
    #Protein BRCA1 phosphorylates protein NBN
    X[1] = np.array([14, 0, 6, 0, 2])
    #Protein MDC1 phosphorylates protein BRCA1
    X[2] = np.array([15, 0, 14, 0, 2])
    #Protein MDC1 is phosphorylated
    X[3] = np.array([0, 0, 15, 0, 2])
    #Protein H2AFX is phosphorylated.
    X[4] = np.array([0, 0, 13, 0, 2])
    print 'Ground truth protein state changes are: '
    print X
    return X


def init_noisy_protein_state_changes():
    n = 5 #number of protein state change observations
    Y = np.zeros([n, 5], np.int)
    # original: Phophorylated protein complex ATM phosphorylates protein TP53
    Y[0] = np.array([6, 3, 1, 0, 2])
    # original: Protein BRCA1 phosphorylates protein NBN
    Y[1] = np.array([14, 0, 6, 0, 2])
    # original: Protein MDC1 phosphorylates protein BRCA1
    Y[2] = np.array([0, 0, 14, 0, 2])
    # original: Protein MDC1 is phosphorylated
    Y[3] = np.array([15, 0, 15, 0, 2])
    # original: Protein H2AFX is phosphorylated
    Y[4] = np.array([13, 0, 15, 0, 2])
    print 'Noisy observations of protein state change are: '
    print Y
    return Y


def filter_invalid_state_changes(X, w):
    n = X.shape[0]
    valid_idx = []
    for i in range(n):
        x = X[i]
        is_valid = True
        if x[0] == x[2]: # and (x[1] == x[3] or x[1] == x[4]): #catalyst and protein can not be same
            is_valid = False
        if x[3] == x[4]: #new and old state of protein can not be same
            is_valid = False
        if x[2] == 0:# or x[0] == 0: #protein can not be None
            is_valid = False
        if x[0] == 0:
            if is_state_change_without_catalyst_allowed:
                w[i] *= 0.8
            else:
                is_valid = False
        if is_valid:
            valid_idx.append(i)
    return X[valid_idx, :], w[valid_idx]


def filter_invalid_complex_forms(X, w):
    n = X.shape[0]
    valid_idx = []
    for i in range(n):
        x = X[i]
        is_valid = True
        #catalyst, protein 1, protein2 can not be same
        if np.unique(x[[0, 2, 4]]).size < 3:
            is_valid = False
        #protein 1, protein2, complex can not be same
        if np.unique(x[[2, 4, 6]]).size < 3:
            is_valid = False
        elif 0 in [x[2], x[4]]: #protein1, protein2 can not be None
            is_valid = False
        if is_valid:
            valid_idx.append(i)
    return X[valid_idx, :], w[valid_idx]


def get_protein_state_changes_and_complex_from_interactions(interactions, protein_names, state_names):
    def get_state_int_frm_str_list(state_list):
        if debug:
            print 'state_list', state_list
        state_vector_bool = np.zeros(shape=state_names.shape, dtype=np.bool)
        if (state_list is not None) and state_list:
            for state in state_list:
                if state is None:
                    raise AssertionError
                state_vector_bool[state_names == state] = True
        num_states = state_vector_bool.size
        return get_state_int_from(np.array([state_vector_bool]))

    def get_protein_int_from_str(protein):
        if protein is None:
            return 0
        else:
            return protein_names.tolist().index(protein)

    if debug:
        print 'protein_names', protein_names
        print 'state_names', state_names
    state_changes = None
    state_changes_weights = None
    if interactions.has_key(state_change) and (interactions[state_change] is not None) and interactions[state_change]:
        num_state_change = len(interactions[state_change])
        state_changes = np.zeros(shape=[num_state_change, 6], dtype=np.int)
        state_changes_weights = []
        for idx, interaction in enumerate(interactions[state_change]):
            state_changes[idx, 0] = get_protein_int_from_str(interaction.catalyst_str)
            state_changes[idx, 1] = get_state_int_frm_str_list(interaction.catalyst_state_str_list)
            state_changes[idx, 2] = get_protein_int_from_str(interaction.protein_str)
            state_changes[idx, 3] = get_state_int_frm_str_list(interaction.protein_state_str_list)
            state_changes[idx, 4] = get_state_int_frm_str_list(interaction.result_state_str_list)
            if not interaction.is_positive_catalyst:
                state_changes[idx, 5] = 0
            else:
                state_changes[idx, 5] = 1
            state_changes_weights.append(interaction.weight)
        state_changes_weights = np.array(state_changes_weights)
        del interaction, num_state_change, idx
        state_changes, unique_idx = remove_duplicates(state_changes, return_idx=True)
        state_changes_weights = state_changes_weights[unique_idx]
        state_changes, state_changes_weights = filter_invalid_state_changes(state_changes, state_changes_weights)
        state_changes, state_changes_weights = sort_interactions_lkl(state_changes, state_changes_weights)
        if state_changes.size == 0:
            state_changes = None
            state_changes_weights = None
    complex_forms = None
    complex_forms_weights = None
    if interactions.has_key(complex) and (interactions[complex] is not None) and interactions[complex]:
        num_complex_forms = len(interactions[complex])
        complex_forms = np.zeros(shape=[num_complex_forms, 9], dtype=np.int)
        complex_forms_weights = []
        for idx, interaction in enumerate(interactions[complex]):
            complex_forms[idx, 0] = get_protein_int_from_str(interaction.catalyst_str)
            complex_forms[idx, 1] = get_state_int_frm_str_list(interaction.catalyst_state_str_list)
            complex_forms[idx, 2] = get_protein_int_from_str(interaction.protein_1_str)
            complex_forms[idx, 3] = get_state_int_frm_str_list(interaction.protein_1_state_str_list)
            complex_forms[idx, 4] = get_protein_int_from_str(interaction.protein_2_str)
            complex_forms[idx, 5] = get_state_int_frm_str_list(interaction.protein_2_state_str_list)
            complex_forms[idx, 6] = get_protein_int_from_str(interaction.complex_str)
            complex_forms[idx, 7] = get_state_int_frm_str_list(interaction.complex_state_str_list)
            if not interaction.is_positive_catalyst:
                complex_forms[idx, 8] = 0
            else:
                complex_forms[idx, 8] = 1
            complex_forms_weights.append(interaction.weight)
        complex_forms_weights = np.array(complex_forms_weights)
        del interaction, num_complex_forms, idx
        complex_forms, unique_idx = remove_duplicates(complex_forms, return_idx=True)
        complex_forms_weights = complex_forms_weights[unique_idx]
        complex_forms, complex_forms_weights = filter_invalid_complex_forms(complex_forms, complex_forms_weights)
        complex_forms, complex_forms_weights = sort_interactions_lkl(complex_forms, complex_forms_weights)
        if complex_forms.size == 0:
            complex_forms = None
            complex_forms_weights = None
    return state_changes, state_changes_weights, complex_forms, complex_forms_weights


def infer_interactions(Y, Y_weights, psi, phi_ijkl, phi_i_jk, phi_s, protein_names, state_names):
    #psi is updated inside this function
    if Y.shape[1] == complex_form_dim:
        is_complex = True
    elif Y.shape[1] == state_change_dim:
        is_complex = False
    else:
        raise AssertionError
    n_y = Y.shape[0]
    print 'No. of protein interactions are: ', n_y
    n = psi.shape[0]
    print 'No. of possible ground truth proteins including NULL are: ', n
    m = psi.shape[1]
    print 'No. of possible ground truth states are: ', m
    for curr_idx in range(n_y):
        y = Y[curr_idx]
        y_weight = Y_weights[curr_idx]
        if len(y) == state_change_dim:
            print 'current noisy interaction is: \n', str(get_state_change_interaction_obj(protein_names, state_names, y))
        elif len(y) == complex_form_dim:
            print 'current noisy interaction is: \n', str(get_complex_form_obj(protein_names, state_names, y))
        if not is_sparse:
            DeprecationWarning
            update_posterior_state_noise_protein_order(y, psi) #psi is updated inside this function
        else:
            update_posterior_state_noise_protein_order_sparse(y, y_weight, psi) #psi is updated inside this function
        psi_sum = psi.sum()
        if psi_sum == 0:
            raise AssertionError
    #todo: as of now, returning empty so as to do inference only at end of processing of all AMRs
    X_inf = np.array([])
    inf_lkl = np.array([])
    return X_inf, inf_lkl


def sort_interactions_lkl(X, lkl):
    #sort the inferences as per the likelihood values in descending order
    print 'lkl:', lkl
    idx_sorted_desc = np.argsort(-lkl) #-inf_lkl will sort in descending
    print 'idx_sorted_desc:', idx_sorted_desc
    X = X[idx_sorted_desc]
    lkl = lkl[idx_sorted_desc]
    return X, lkl


def get_most_lkl_interactions(psi, m=None):
    if m is None:
        print 'psi.shape:', psi.shape
        x = psi.flatten()
        print 'x:', x
        y = np.where(x > min_posterior_lkl)
    elif m < 1:
        raise AssertionError
    else:
        #m is number of most likely interactions
        print 'm:', m
        print 'psi.shape:', psi.shape
        x = psi.flatten()
        print 'x:', x
        y = heapq.nlargest(m, range(len(x)), x.take)
    inf_lkl = x[y]
    x = None
    del x
    print 'inf_lkl:', inf_lkl
    y = y[0].tolist()
    print 'y: ', y
    print 'psi.shape:', psi.shape
    idx_tuples = np.unravel_index(y, psi.shape)
    y = None
    del y
    print 'idx_tuples:', idx_tuples
    X_inf = np.array(idx_tuples).transpose()
    idx_tuples = None
    del idx_tuples
    #remove duplicates and then sort as per lkl since the duplication removal unsort somehow
    X_inf, unique_idx = remove_duplicates(X_inf, return_idx=True)
    inf_lkl = inf_lkl[unique_idx]
    X_inf, inf_lkl = filter_inferred(X_inf, inf_lkl)
    X_inf, inf_lkl = sort_interactions_lkl(X_inf, inf_lkl)
    return X_inf, inf_lkl


def filter_out_less_likely_versions(X, Y, protein_names, state_names):
    #X is interactions (2-D array)
    #Y is likelihood values
    #this function assumes that the interactions are sorted in decreasing likelihood order
    #todo: add functionality to check lkl vector is sorted (which will indirectly ensure that X is also sorted)
    #it is also assumed that duplicates are removed
    n = X.shape[0]
    if X.shape[1] == state_change_dim:
        is_state_change = True
    elif X.shape[1] == complex_form_dim:
        is_state_change = False
    else:
        raise AssertionError
    X = X.tolist()
    Y = Y.tolist()
    #store likelihood values in a map
    lkl_map = {}
    for i in range(n):
        lkl_map[tuple(X[i])] = Y[i]
    X_copy = copy.deepcopy(X)
    for i in range(n):
        xi = X_copy[i]
        # if is_state_change:
        #     xi_obj = get_state_change_interaction_obj(protein_names, state_names, xi, lkl_map[tuple(xi)])
        # else:
        #     xi_obj = get_complex_form_obj(protein_names, state_names, xi, lkl_map[tuple(xi)])
        if xi in X:
            for j in range(i+1, n):
                xj = X_copy[j]
                if is_state_change:
                    xj_obj = get_state_change_interaction_obj(protein_names, state_names, xj, lkl_map[tuple(xj)])
                # else:
                #     xj_obj = get_complex_form_obj(protein_names, state_names, xj, lkl_map[tuple(xj)])
                if xj in X:
                    if is_state_change:
                        if xi[0] == 0 or xj[0] == 0:
                            xi_sub = (xi[2])
                            xj_sub = (xj[2])
                        xi_sub = (xi[0], xi[2])
                        xj_sub = (xj[0], xj[2])
                    else:
                        xi_sub = (xi[0], xi[2], xi[4])
                        xj_sub = (xj[0], xj[2], xj[4])
                    if xi_sub == xj_sub and lkl_map[tuple(xj)] < lkl_map[tuple(xi)]:
                        if is_state_change:
                            if xi[0] == 0 or xj[0] == 0:
                                if not xj_obj.is_mechanistic_information():
                                    if xj[0] == 0:
                                        if xj in X:
                                            X.remove(xj)
                                    elif xi[0] == 0:
                                        if xi in X:
                                            X.remove(xi)
                                else:
                                    if (xi[3], xi[4]) == (xj[3], xj[4]):
                                        if xj[0] == 0:
                                            if xj in X:
                                                X.remove(xj)
                                        elif xi[0] == 0:
                                            if xi in X:
                                                X.remove(xi)
                            else:
                                if not xj_obj.is_mechanistic_information():
                                    if xj in X:
                                        X.remove(xj)
                                else:
                                    if (xi[3], xi[4]) == (xj[3], xj[4]):
                                        if xj in X:
                                            X.remove(xj)
                        else:
                            if xj in X:
                                X.remove(xj)
    #get the corresponding likelihood values from the map,
    # and also ensure that all similar interactions with equal likelihood are together in the list/array
    X_new = []
    Y_new = []
    n = len(X)
    for i in range(n):
        xi = X[i]
        if xi not in X_new:
            X_new.append(xi)
            Y_new.append(lkl_map[tuple(xi)])
            for j in range(i+1, n):
                xj = X[j]
                if xj not in X_new:
                    if is_state_change:
                        xi_sub = (xi[0], xi[2])
                        xj_sub = (xj[0], xj[2])
                    else:
                        xi_sub = (xi[0], xi[2], xi[4])
                        xj_sub = (xj[0], xj[2], xj[4])
                    if xi_sub == xj_sub: #similar interaction
                        # if lkl_map[tuple(xj)] != lkl_map[tuple(xi)]:
                        #     raise AssertionError
                        X_new.append(xj)
                        Y_new.append(lkl_map[tuple(xj)])
    X = np.array(X_new)
    Y = np.array(Y_new)
    return X, Y


def filter_inferred(X_inf, inf_lkl):
    if X_inf.shape[1] == state_change_dim:
        X_inf, inf_lkl = filter_invalid_state_changes(X_inf, inf_lkl)
    elif X_inf.shape[1] == complex_form_dim:
        X_inf, inf_lkl = filter_invalid_complex_forms(X_inf, inf_lkl)
    else:
        raise AssertionError
    return X_inf, inf_lkl


def get_most_lkl_interactions_sparse(psi, psi_init, protein_names, state_names):
    if min_posterior_lkl < psi.default:
        raise AssertionError('Too many interactions ...') #assuming self as minimal value for efficiency
    else:
        X_inf = []
        inf_lkl = []
        for k in psi.data.keys():
            val = psi[k]
            # print 'val:', val
            if val > min_posterior_lkl:
                if psi[k] != psi_init[k]:
                    X_inf.append(k)
                    inf_lkl.append(val)
    X_inf = np.array(X_inf)
    print 'X_inf:', X_inf
    inf_lkl = np.array(inf_lkl)
    print 'inf_lkl:', inf_lkl
    X_inf, unique_idx = remove_duplicates(X_inf, return_idx=True)
    inf_lkl = inf_lkl[unique_idx]
    X_inf, inf_lkl = filter_inferred(X_inf, inf_lkl)
    X_inf, inf_lkl = sort_interactions_lkl(X_inf, inf_lkl)
    if cgp.is_filter_out_less_likely_versions:
        X_inf, inf_lkl = filter_out_less_likely_versions(X_inf, inf_lkl, protein_names, state_names)
        X_inf, inf_lkl = sort_interactions_lkl(X_inf, inf_lkl)
    return X_inf, inf_lkl


def marginal_prob_fr_sum_min_one(t):
    #noise likelihood specifies that, with 0.1 probability one of other candidate interactions (besides the observed one) is one.
    #since this is a joint probability, we take marginals
    #marginal probability for all these variables is equal because of the type of constraint
    marginal_prob = 1/float(2 - 2**(-t+1))
    return marginal_prob


def update_noise_lkl(y, psi, lkl, idx_bool):
    raise NotImplementedError
    #idx_bool represents which positions in an interaction to be noised
    # #misplace of catalyst protein
    # for i in range(n).remove(y[0]):
    #     x = np.copy(y) #candidate interaction
    #     x[0]


def eval_lkl_naive_fr_state_change_interaction(y, psi):
    raise NotImplementedError
    n = psi.shape[0] #no. of proteins
    m = psi.shape[1] #no. of states
    prob_obs_org = 0.9 #probability of observing protein or state
    lkl = np.ones(psi.shape)
    #observing same interaction, no marginal required
    lkl[tuple(y)] = prob_obs_org
    #cases where one of {proteins, state is misplaced}
    #first protein misplace case
    psi *= lkl #*0.999 #adding little noise to make posterior values unequal
    psi_sum = psi.sum()
    if psi_sum != 0:
        psi /= psi_sum
    return psi


def update_posterior_base(y, psi):
    psi_sum = psi.sum()
    prob_obs_org = 0.8 #probability of observing original interaction
    y = tuple(y)
    print 'y:', y
    psi_y = psi[y]
    print 'psi_y is:', psi_y
    prob_all_zero_interactions = (np.prod(1-psi)/float(1-psi_y)) #all interactions zero except the observed interaction y
    print 'prob_all_zero_interactions:', prob_all_zero_interactions
    #since it is bernoulli trial of each interaction, marginal for each candidate interaction is different if their posterior distribution is different
    #evaluation joint lkl and then division by marginal to get posterior
    psi *= (prob_obs_org*psi_y + (1-prob_obs_org)*(1-psi_y)) #joint likelihood
    marginal_except_y = (prob_obs_org*psi_y) + ((1-prob_obs_org)*(1-psi_y)*(1-prob_all_zero_interactions)) #marginal for all except y interaction
    print 'marginal_except_y:', marginal_except_y
    psi /= marginal_except_y #division by marginal
    #mind that we need to overwrite the previous posterior assignment for y (the assignment was done just for efficient assignment for rest of the candidate interaction)
    psi[y] = psi_y*prob_obs_org
    print 'posterior[y] not normalized:', psi[y]
    marginal_y = (1-prob_obs_org)*(1 -psi_y -((1-psi_y)*prob_all_zero_interactions)) + psi_y*prob_obs_org
    print 'marginal_y:', marginal_y
    psi[y] /= marginal_y
    print 'posterior[y] normalized:', psi[y]
    print 'psi sum is', psi_sum
    print 'posterior sum is', psi.sum()
    return psi


def update_posterior_state_noise_protein_order_sparse(y, y_weight, psi):
    def update_prior_all_one_state_change(psi, mesh_idx, prior_all_one_state_change):
        # print 'mesh_idx:', mesh_idx
        expr = (1-psi[mesh_idx])
        expr_ones = 1-expr
        # print 'expr:', expr
        # print 'type(expr):', type(expr)
        if type(expr) == psi.__class__:
            expr = expr.prod()
            expr_ones = expr_ones.prod()
        elif not isinstance(expr, (int, long, float)):
            raise AssertionError
        prior_all_one_state_change *= expr
        print 'prior_all_one_state_change:', prior_all_one_state_change
        return prior_all_one_state_change

    #probability in noise model
    prob_obs_org = 0.7 #probability of observing original interaction
    prob_obs_one_state_change = 0.15
    prob_obs_protein_order_change = 0.15
    #get the observed interaction (evidence)
    y = tuple(y)
    print 'y:', y
    #prior for original interaction
    psi_y = psi[y]
    #evaluate probability for all interactions zero with one state change
    prior_all_one_state_change = 1
    prior_all_one_state_change_ones = 1
    m = psi.shape[1] #no. of states
    psi_shape = psi.shape
    if len(psi_shape) == state_change_dim:
        #interactions with catalyst state noise
        other_states = np.setdiff1d(np.arange(m), np.array([y[1]]))
        mesh_idx = (np.array([y[0]]), other_states, np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]))
        prior_all_one_state_change = update_prior_all_one_state_change(psi, mesh_idx, prior_all_one_state_change)
        #interactions with protein state noise
        other_states = np.setdiff1d(np.arange(m), np.array([y[3]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), other_states, np.array([y[4]]), np.array([y[5]]))
        prior_all_one_state_change = update_prior_all_one_state_change(psi, mesh_idx, prior_all_one_state_change)
        #interactions with protein new state noise
        other_states = np.setdiff1d(np.arange(m), np.array([y[4]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), other_states, np.array([y[5]]))
        prior_all_one_state_change = update_prior_all_one_state_change(psi, mesh_idx, prior_all_one_state_change)
    elif len(psi_shape) == complex_form_dim:
        #catalyst state noise
        other_states = np.setdiff1d(np.arange(m), np.array([y[1]]))
        mesh_idx = (np.array([y[0]]), other_states, np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), np.array([y[7]]), np.array([y[8]]))
        prior_all_one_state_change = update_prior_all_one_state_change(psi, mesh_idx, prior_all_one_state_change)
        #protein 1 state noise
        other_states = np.setdiff1d(np.arange(m), np.array([y[3]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), other_states, np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), np.array([y[7]]), np.array([y[8]]))
        prior_all_one_state_change = update_prior_all_one_state_change(psi, mesh_idx, prior_all_one_state_change)
        #protein 2 state noise
        other_states = np.setdiff1d(np.arange(m), np.array([y[5]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), other_states, np.array([y[6]]), np.array([y[7]]), np.array([y[8]]))
        prior_all_one_state_change = update_prior_all_one_state_change(psi, mesh_idx, prior_all_one_state_change)
        #complex state noise
        other_states = np.setdiff1d(np.arange(m), np.array([y[7]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), other_states, np.array([y[8]]))
        prior_all_one_state_change = update_prior_all_one_state_change(psi, mesh_idx, prior_all_one_state_change)
    else:
        raise AssertionError
    print 'prior_all_one_state_change is: ', prior_all_one_state_change
    prior_all_protein_order_change = 1
    if len(psi_shape) == state_change_dim:
        for i, j in itertools.permutations(np.array([0, 2]), 2):
            if not(i == 0 and j == 2):
                x_tuple = (y[i], y[1], y[j], y[3], y[4], y[5])
                prior_all_protein_order_change *= (1-psi[x_tuple])
    elif len(psi_shape) == complex_form_dim:
        X = []
        for i, j, k, l in itertools.permutations(np.array([0, 2, 4, 6]), 4):
            if not(i == 0 and j == 2 and k == 4 and l == 6):
                x_tuple = (y[i], y[1], y[j], y[3], y[k], y[5], y[l], y[7], y[8])
                x = list(x_tuple)
                if x not in X:
                    X.append(x)
                    prior_all_protein_order_change *= (1-psi[x_tuple])
        X = None
    else:
        raise AssertionError
    print 'prior_all_protein_order_change:', prior_all_protein_order_change
    #since it is bernoulli trial of each interaction, marginal for each candidate interaction is different if their posterior distribution is different
    #evaluation joint lkl and then division by marginal to get posterior
    ##evaluate posterior for evidence interaction
    psi[y] = prob_obs_org*psi_y
    print 'posterior[y] before marginalization:', psi[y]
    #normalize
    marginal_y = psi[y] + prob_obs_one_state_change*(1-prior_all_one_state_change)*(1-psi_y) + prob_obs_protein_order_change*(1-prior_all_protein_order_change)*(1-psi_y)
    print 'marginal_y:', marginal_y
    psi[y] /= 1e-300 + marginal_y
    print 'posterior[y] after marginalization:', psi[y]
    if cgp.is_sample_weight_in_bayesian_update:
        psi[y] = psi[y]*y_weight + (1-y_weight)*psi_y
        print 'posterior[y] after accounting for weight of the observed sample:', psi[y]
    #evaluating posterior for interaction candidates with one state change
    if len(psi_shape) == state_change_dim:
        #
        other_states = np.setdiff1d(np.arange(m), np.array([y[1]]))
        mesh_idx = (np.array([y[0]]), other_states, np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]))
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        if cgp.is_sample_weight_in_bayesian_update:
            psi[mesh_idx] = psi[mesh_idx]*y_weight + prior*(1-y_weight)
        #
        other_states = np.setdiff1d(np.arange(m), np.array([y[3]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), other_states, np.array([y[4]]), np.array([y[5]]))
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        if cgp.is_sample_weight_in_bayesian_update:
            psi[mesh_idx] = psi[mesh_idx]*y_weight + prior*(1-y_weight)
        #
        other_states = np.setdiff1d(np.arange(m), np.array([y[4]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), other_states, np.array([y[5]]))
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        if cgp.is_sample_weight_in_bayesian_update:
            psi[mesh_idx] = psi[mesh_idx]*y_weight + prior*(1-y_weight)
    elif len(psi_shape) == complex_form_dim:
        #catalyst state
        other_states = np.setdiff1d(np.arange(m), np.array([y[1]]))
        mesh_idx = (np.array([y[0]]), other_states, np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), np.array([y[7]]), np.array([y[8]]))
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        if cgp.is_sample_weight_in_bayesian_update:
            psi[mesh_idx] = psi[mesh_idx]*y_weight + prior*(1-y_weight)
        #protein 1 state
        other_states = np.setdiff1d(np.arange(m), np.array([y[3]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), other_states, np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), np.array([y[7]]), np.array([y[8]]))
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        if cgp.is_sample_weight_in_bayesian_update:
            psi[mesh_idx] = psi[mesh_idx]*y_weight + prior*(1-y_weight)
        #protein 2 state
        other_states = np.setdiff1d(np.arange(m), np.array([y[5]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), other_states, np.array([y[6]]), np.array([y[7]]), np.array([y[8]]))
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        if cgp.is_sample_weight_in_bayesian_update:
            psi[mesh_idx] = psi[mesh_idx]*y_weight + prior*(1-y_weight)
        #complex state
        other_states = np.setdiff1d(np.arange(m), np.array([y[7]]))
        mesh_idx = (np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), other_states, np.array([y[8]]))
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        if cgp.is_sample_weight_in_bayesian_update:
            psi[mesh_idx] = psi[mesh_idx]*y_weight + prior*(1-y_weight)
    else:
        raise AssertionError
    print 'updating posterior for interactions with protein order change'
    if len(psi_shape) == state_change_dim:
        X = []
        for i, j in itertools.permutations(np.array([0, 2]), 2):
            if not(i == 0 and j == 2):
                x_tuple = (y[i], y[1], y[j], y[3], y[4], y[5])
                x = list(x_tuple)
                if x not in X:
                    X.append(x)
                    print 'x_tuple:', x_tuple
                    prior = psi[x_tuple]
                    print 'prior:', prior
                    psi[x_tuple] = prior*(prob_obs_org*psi_y + (1-psi_y)*(prob_obs_one_state_change*(1-prior_all_one_state_change) + prob_obs_protein_order_change))
                    print 'posterior before marginalization:', psi[x_tuple]
                    marginal = psi[x_tuple] + prob_obs_org*psi_y*(1-prior) + (1-psi_y)*(prob_obs_one_state_change*(1-prior)*(1-prior_all_one_state_change) + prob_obs_protein_order_change*((1-prior) - prior_all_protein_order_change))
                    psi[x_tuple] /= 1e-300 + marginal
                    print 'posterior after marginalization:', psi[x_tuple]
                    if cgp.is_sample_weight_in_bayesian_update:
                        psi[x_tuple] = psi[x_tuple]*y_weight + prior*(1-y_weight)
                        print 'posterior[y] after accounting for weight of the observed sample:', psi[x_tuple]
    elif len(psi_shape) == complex_form_dim:
        X = []
        for i, j, k, l in itertools.permutations(np.array([0, 2, 4, 6]), 4):
            if not(i == 0 and j == 2 and k == 4 and l == 6):
                x_tuple = (y[i], y[1], y[j], y[3], y[k], y[5], y[l], y[7], y[8])
                x = list(x_tuple)
                if x not in X:
                    X.append(x)
                    print 'x_tuple:', x_tuple
                    prior = psi[x_tuple]
                    print 'prior:', prior
                    psi[x_tuple] = prior*(prob_obs_org*psi_y + (1-psi_y)*(prob_obs_one_state_change*(1-prior_all_one_state_change) + prob_obs_protein_order_change))
                    print 'posterior before marginalization:', psi[x_tuple]
                    marginal = psi[x_tuple] + prob_obs_org*psi_y*(1-prior) + (1-psi_y)*(prob_obs_one_state_change*(1-prior)*(1-prior_all_one_state_change) + prob_obs_protein_order_change*((1-prior) - prior_all_protein_order_change))
                    print 'marginal:', marginal
                    psi[x_tuple] /= 1e-300 + marginal
                    print 'posterior after marginalization:', psi[x_tuple]
                    if cgp.is_sample_weight_in_bayesian_update:
                        psi[x_tuple] = psi[x_tuple]*y_weight + prior*(1-y_weight)
                        print 'posterior[y] after accounting for weight of the observed sample:', psi[x_tuple]
        X = None
    else:
        raise AssertionError


def update_posterior_state_noise_protein_order(y, psi):
    # #if evidence interaction is not positive, the psi was changed to 1-psi before inference and changed back to 1-psi after the inference
    # #1-psi basically represent the probability values for negative interactions while psi represent for positive. the 1-psi operation before inference and after inference gives us psi back i.e. prior/posterior for positive interactions
    # if not y_pos:
    #     psi = 1-psi
    #prior sum backup for later comparison (logging)
    psi_sum = psi.sum()
    #probability in noise model
    prob_obs_org = 0.7 #probability of observing original interaction
    prob_obs_one_state_change = 0.15
    # prob_obs_more_state_change = 0.1
    prob_obs_protein_order_change = 0.15
    #get the observed interaction (evidence)
    y = tuple(y)
    print 'y:', y
    #prior for original interaction
    psi_y = psi[y]
    print 'psi_y is:', psi_y
    #evaluate probability for all interactions zero with one state change
    prior_all_one_state_change = 1
    m = psi.shape[1] #no. of states
    psi_shape = psi.shape
    # candidates = np.zeros(shape=psi.shape, dtype=np.bool) #for rest of the interactions, it becomes 0.5
    #state_pos = np.array([1, 3, 4]) state change
    #state_pos = np.array([1, 3, 5, 7]) complex formation
    if len(psi_shape) == 5:
        other_states = np.setdiff1d(np.arange(m), np.array([y[1]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), other_states, np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), indexing='ij')
        # candidates[mesh_idx] = True
        prior_all_one_state_change *= np.prod(1-psi[mesh_idx])
        other_states = np.setdiff1d(np.arange(m), np.array([y[3]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), other_states, np.array([y[4]]), indexing='ij')
        # candidates[mesh_idx] = True
        prior_all_one_state_change *= np.prod(1-psi[mesh_idx])
        other_states = np.setdiff1d(np.arange(m), np.array([y[4]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), other_states, indexing='ij')
        # candidates[mesh_idx] = True
        prior_all_one_state_change *= np.prod(1-psi[mesh_idx])
    elif len(psi_shape) == 8:
        #catalyst state
        other_states = np.setdiff1d(np.arange(m), np.array([y[1]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), other_states, np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), np.array([y[7]]), indexing='ij')
        # candidates[mesh_idx] = True
        prior_all_one_state_change *= np.prod(1-psi[mesh_idx])
        #protein 1 state
        other_states = np.setdiff1d(np.arange(m), np.array([y[3]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), other_states, np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), np.array([y[7]]), indexing='ij')
        # candidates[mesh_idx] = True
        prior_all_one_state_change *= np.prod(1-psi[mesh_idx])
        #protein 2 state
        other_states = np.setdiff1d(np.arange(m), np.array([y[5]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), other_states, np.array([y[6]]), np.array([y[7]]), indexing='ij')
        # candidates[mesh_idx] = True
        prior_all_one_state_change *= np.prod(1-psi[mesh_idx])
        #complex state
        other_states = np.setdiff1d(np.arange(m), np.array([y[7]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), other_states, indexing='ij')
        # candidates[mesh_idx] = True
        prior_all_one_state_change *= np.prod(1-psi[mesh_idx])
    else:
        raise AssertionError
    print 'prior_all_one_state_change is: ', prior_all_one_state_change
    prior_all_protein_order_change = 1
    if len(psi_shape) == 5:
        for i, j in itertools.permutations(np.array([0, 2]), 2):
            if not(i == 0 and j == 2):
                x_tuple = (y[i], y[1], y[j], y[3], y[4])
                # candidates[x_tuple] = True
                prior_all_protein_order_change *= (1-psi[x_tuple])
    elif len(psi_shape) == 8:
        X = []
        for i, j, k, l in itertools.permutations(np.array([0, 2, 4, 6]), 4):
            if not(i == 0 and j == 2 and k == 4 and l == 6):
                x_tuple = (y[i], y[1], y[j], y[3], y[k], y[5], y[l], y[7])
                x = list(x_tuple)
                if x not in X:
                    X.append(x)
                    # candidates[x_tuple] = True
                    prior_all_protein_order_change *= (1-psi[x_tuple])
        X = None
    else:
        raise AssertionError
    print 'prior_all_protein_order_change:', prior_all_protein_order_change
    #since it is bernoulli trial of each interaction, marginal for each candidate interaction is different if their posterior distribution is different
    #evaluation joint lkl and then division by marginal to get posterior
    ##evaluate posterior for evidence interaction
    psi[y] = prob_obs_org*psi_y
    print 'posterior[y] before marginalization:', psi[y]
    #normalize
    marginal_y = psi[y] + prob_obs_one_state_change*(1-prior_all_one_state_change)*(1-psi_y) + prob_obs_protein_order_change*(1-prior_all_protein_order_change)*(1-psi_y)
    print 'marginal_y:', marginal_y
    psi[y] /= 1e-300 + marginal_y
    print 'posterior[y] after marginalization:', psi[y]
    #evaluating posterior for interaction candidates with one state change
    if len(psi_shape) == 5:
        other_states = np.setdiff1d(np.arange(m), np.array([y[1]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), other_states, np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), indexing='ij')
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        other_states = np.setdiff1d(np.arange(m), np.array([y[3]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), other_states, np.array([y[4]]), indexing='ij')
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        other_states = np.setdiff1d(np.arange(m), np.array([y[4]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), other_states, indexing='ij')
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
    elif len(psi_shape) == 8:
        #catalyst state
        other_states = np.setdiff1d(np.arange(m), np.array([y[1]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), other_states, np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), np.array([y[7]]), indexing='ij')
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        #protein 1 state
        other_states = np.setdiff1d(np.arange(m), np.array([y[3]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), other_states, np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), np.array([y[7]]), indexing='ij')
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        #protein 2 state
        other_states = np.setdiff1d(np.arange(m), np.array([y[5]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), other_states, np.array([y[6]]), np.array([y[7]]), indexing='ij')
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
        #complex state
        other_states = np.setdiff1d(np.arange(m), np.array([y[7]]))
        mesh_idx = np.meshgrid(np.array([y[0]]), np.array([y[1]]), np.array([y[2]]), np.array([y[3]]), np.array([y[4]]), np.array([y[5]]), np.array([y[6]]), other_states, indexing='ij')
        prior = psi[mesh_idx]
        psi[mesh_idx] = prior*(prob_obs_org*psi_y + prob_obs_one_state_change*(1-psi_y) + prob_obs_protein_order_change*(1-psi_y)*(1-prior_all_protein_order_change))
        marginal = psi[mesh_idx] + prob_obs_org*psi_y*(1-prior) + prob_obs_one_state_change*(1-psi_y)*((1-prior) - prior_all_one_state_change) + prob_obs_protein_order_change*(1-psi_y)*(1-prior)*(1-prior_all_protein_order_change)
        psi[mesh_idx] /= 1e-300 + marginal
    else:
        raise AssertionError
    print 'updating posterior for interactions with protein order change'
    if len(psi_shape) == 5:
        X = []
        for i, j in itertools.permutations(np.array([0, 2]), 2):
            if not(i == 0 and j == 2):
                x_tuple = (y[i], y[1], y[j], y[3], y[4])
                x = list(x_tuple)
                if x not in X:
                    X.append(x)
                    print 'x_tuple:', x_tuple
                    prior = psi[x_tuple]
                    print 'prior:', prior
                    psi[x_tuple] = prior*(prob_obs_org*psi_y + (1-psi_y)*(prob_obs_one_state_change*(1-prior_all_one_state_change) + prob_obs_protein_order_change))
                    print 'posterior before marginalization:', psi[x_tuple]
                    marginal = psi[x_tuple] + prob_obs_org*psi_y*(1-prior) + (1-psi_y)*(prob_obs_one_state_change*(1-prior)*(1-prior_all_one_state_change) + prob_obs_protein_order_change*((1-prior) - prior_all_protein_order_change))
                    psi[x_tuple] /= 1e-300 + marginal
                    print 'posterior after marginalization:', psi[x_tuple]
    elif len(psi_shape) == 8:
        X = []
        for i, j, k, l in itertools.permutations(np.array([0, 2, 4, 6]), 4):
            if not(i == 0 and j == 2 and k == 4 and l == 6):
                x_tuple = (y[i], y[1], y[j], y[3], y[k], y[5], y[l], y[7])
                x = list(x_tuple)
                if x not in X:
                    X.append(x)
                    print 'x_tuple:', x_tuple
                    prior = psi[x_tuple]
                    print 'prior:', prior
                    psi[x_tuple] = prior*(prob_obs_org*psi_y + (1-psi_y)*(prob_obs_one_state_change*(1-prior_all_one_state_change) + prob_obs_protein_order_change))
                    print 'posterior before marginalization:', psi[x_tuple]
                    marginal = psi[x_tuple] + prob_obs_org*psi_y*(1-prior) + (1-psi_y)*(prob_obs_one_state_change*(1-prior)*(1-prior_all_one_state_change) + prob_obs_protein_order_change*((1-prior) - prior_all_protein_order_change))
                    print 'marginal:', marginal
                    psi[x_tuple] /= 1e-300 + marginal
                    print 'posterior after marginalization:', psi[x_tuple]
        X = None
    else:
        raise AssertionError
    print 'psi sum is', psi_sum
    print 'posterior sum is', psi.sum()
    # #if evidence interaction is not positive, the psi was changed to 1-psi before inference and changed back to 1-psi after the inference
    # #1-psi basically represent the probability values for negative interactions while psi represent for positive. the 1-psi operation before inference and after inference gives us psi back i.e. prior/posterior for positive interactions
    # if not y_pos:
    #     psi = 1-psi


def eval_lkl(X, y, psi, phi_ijkl, phi_i_jk, phi_s):
    #psi is also updated as per the posterior inhere
    n_x = X.shape[0]
    if X.shape[1] == 8:
        is_complex = True
    elif X.shape[1] == 5:
        is_complex = False
    else:
        raise AssertionError
    # lkl = np.zeros(n_x)
    lkl = 0.1*np.ones(psi.shape) #posterior is zero for rest of interactions since X is set of all possible interactions
    #todo: remove for loop and vectorize the operation
    for curr_idx in np.arange(n_x):
        x = X[curr_idx]
        #p(x)
        prior_model_lkl = psi[tuple(x)]
        #P(y|x)
        #swap
        noise_lkl_1 = phi_i_jk[x[0], x[2], y[0], y[2]]
        if is_complex:
            noise_lkl_1 *= phi_i_jk[x[0], x[4], y[0], y[4]]
        noise_lkl_2 = phi_ijkl[x[0], y[0]]*phi_ijkl[x[2], y[2]]
        if is_complex:
            noise_lkl_2 *= phi_ijkl[x[4], y[4]]*phi_ijkl[x[6], y[6]]
        if not is_complex: #state change interaction
            state_noise_lkl = phi_s[x[1], y[1]]*phi_s[x[3], y[3]]*phi_s[x[4], y[4]]
        else:
            state_noise_lkl = phi_s[x[1], y[1]]*phi_s[x[3], y[3]]*phi_s[x[5], y[5]]*phi_s[x[7], y[7]]
        # noise_lkl = 0.5*(noise_lkl_1 + noise_lkl_2)*state_noise_lkl
        noise_lkl = 0.5*noise_lkl_2*state_noise_lkl
        #p(x,y)
        # lkl[curr_idx] = prior_model_lkl*noise_lkl
        #updating posterior as prior
        lkl[tuple(x)] = noise_lkl
    # print 'likelihood for possible ground truth protein state changes are: '
    # print lkl
    psi *= lkl #*0.999 #adding little noise to make posterior values unequal
    psi_sum = psi.sum()
    if psi_sum != 0:
        psi /= psi_sum
    return psi


def generate_possible_ground_truth_protein_state_changes(y, n, m):
    #y is observed protein state change interaction
    #n is no. of possible ground truth proteins including NULL
    n_x = (n**2)*(m**3)
    X = np.zeros([n_x, 5], np.int)
    count = 0
    for i in np.arange(n): #catalyst protein index
        for j in np.arange(n): #NULL protein can not be present at position j in protein state change ground truth
            for s_i in np.arange(m):
                for s_j in np.arange(m):
                    for s_j_new in np.arange(m):
                        X[count] = np.copy(y)
                        X[count, 0] = i
                        X[count, 1] = s_i
                        X[count, 2] = j
                        X[count, 3] = s_j
                        X[count, 4] = s_j_new
                        count += 1
    # print 'Given observed protein state change (y): ', y
    # print 'Possible ground state changes are: '
    # print X
    return X


def generate_possible_ground_truth_complex_forms(y, n, m):
    #y is observed complex formation interaction
    #n is no. of possible ground truth proteins including NULL
    n_x = (n**4)*(m**4)
    X = np.zeros([n_x, 8], np.int)
    count = 0
    for i in np.arange(n): #catalyst protein index
        for j in np.arange(n): #NULL protein can be present at position j in complex
            for k in np.arange(n): #NULL protein can be present at position k in complex
                for l in np.arange(n): #NULL protein can be present at position l in complex
                    for s_i in np.arange(m):
                        for s_j in np.arange(m):
                            for s_k in np.arange(m):
                                for s_l in np.arange(m):
                                    X[count] = np.copy(y)
                                    X[count, 0] = i
                                    X[count, 1] = s_i
                                    X[count, 2] = j
                                    X[count, 3] = s_j
                                    X[count, 4] = k
                                    X[count, 5] = s_k
                                    X[count, 6] = l
                                    X[count, 7] = s_l
                                    count += 1
    # print 'Given observed protein state change (y): ', y
    # print 'Possible ground state changes are: '
    # print X
    return X


def remove_duplicates(X, return_idx=False):
    if len(X.shape) != 2:
        raise NotImplementedError
    #source: http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
    #though logic conceptually understood
    X = np.ascontiguousarray(X)
    view_X = X.view([('', X.dtype)]*X.shape[1])
    if return_idx:
        unique_X, unique_idx = np.unique(view_X, return_index=return_idx)
    else:
        unique_X = np.unique(view_X)
    del view_X
    result_X = unique_X.view(X.dtype).reshape((unique_X.shape[0], X.shape[1]))
    del X, unique_X
    if return_idx:
        return result_X, unique_idx
    else:
        return result_X


def write_owl(X_s, X_c, protein_names, state_names, file_path):
    write_owl_header(file_path)
    write_sequence_modifications_owl(file_path)
    write_all_protein_complex_owl(X_s, X_c, protein_names, state_names, file_path)
    write_protein_complex_formations_owl(X_c, protein_names, state_names, file_path)
    write_protein_state_change_owl(X_s, protein_names, state_names, file_path)
    write_owl_footer(file_path)


def write_owl_header(file_path):
    f = open(cap.absolute_path+file_path, 'w')
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<rdf:RDF\n')
    f.write(' xmlns:xsd="http://www.w3.org/2001/XMLSchema#"\n')
    f.write(' xmlns:owl="http://www.w3.org/2002/07/owl#"\n')
    f.write(' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n')
    f.write(' xmlns:bp="http://www.biopax.org/release/biopax-level3.owl#"\n')
    f.write(' xml:base="http://purl.org/pc2/4/">\n')
    f.write('<owl:Ontology rdf:about="">\n')
    f.write(' <owl:imports rdf:resource="http://www.biopax.org/release/biopax-level3.owl#" />\n')
    f.write('</owl:Ontology>\n')
    f.close()


def write_owl_footer(file_path):
    f = open(cap.absolute_path+file_path, 'a')
    f.write('</rdf:RDF>\n')
    f.close()


def write_sequence_modifications_owl(file_path):
    file_str = ''
    for state in ['phosphoserine', 'active', 'inactive']:
        state = str(state) #remove single quotes in case
        smv_id = 'SMV_'+state
        file_str += '<bp:SequenceModificationVocabulary rdf:ID="{}">\n'.format(smv_id)
        file_str += '<bp:term rdf:datatype = "http://www.w3.org/2001/XMLSchema#string">{}</bp:term>\n'.format(state)
        file_str += '</bp:SequenceModificationVocabulary>\n'
        mf_id = 'MF_'+state
        file_str += '<bp:ModificationFeature rdf:ID="{}">\n'.format(mf_id)
        file_str += '<bp:modificationType rdf:resource="#{}" />\n'.format(smv_id)
        file_str += '</bp:ModificationFeature>\n'
    f = open(cap.absolute_path+file_path, 'a')
    f.write(file_str)
    f.close()


def get_feature_str_owl(state):
    if state is not None:
        mf_ref_str = 'MF_'
        if 'phospho' in state:
            mf_ref_str += phosphoserine
        elif 'activ' in state:
            mf_ref_str += active
        elif 'inactiv' in state:
            mf_ref_str += inactive
        # else:
        #     raise AssertionError
        return '<bp:feature rdf:resource="#{}" />\n'.format(mf_ref_str)


def write_all_protein_complex_owl(X_s, X_c, protein_names, state_names, file_path):
    k = state_names.size

    def get_str_fr_complex(complex_id, complex_name, complex_state_int=None, component_ids=None):
        complex_str = ''
        if complex_name is not None and complex_name != NULL:
            complex_str = '<bp:Complex rdf:ID="{}">\n'
            complex_str = complex_str.format(complex_id)
            complex_str += '<bp:displayName rdf:datatype = "http://www.w3.org/2001/XMLSchema#string">{}</bp:displayName>\n'
            complex_str = complex_str.format(complex_name)
            if component_ids is not None:
                components_str = ''
                for component_id in component_ids:
                    components_str += '<bp:component rdf:resource="#{}" />'.format(component_id)
                complex_str += components_str
            #add states
            #complex with components do not show its own state
            if (component_ids is None or not component_ids) and complex_state_int is not None:
                states = get_state_string_vector_from(get_state_vector_from(complex_state_int, k), state_names)
                for state in states:
                    complex_str += get_feature_str_owl(state)
            complex_str += '</bp:Complex>\n'
        return complex_str

    #X_s is an array of interactions related to protein state change
    #X_c is an array of interactions related to complex formations
    file_str = ''
    complex_added_list = []
    if X_s is not None:
        n_s = X_s.shape[0]
        for curr_idx in range(n_s):
            x = X_s[curr_idx]
            catalyst_name = protein_names[x[0]]
            catalyst_id = str(x[0])+str(x[1])
            state_change_protein_name = protein_names[x[2]]
            state_change_protein_id = str(x[2])+str(x[3])
            new_state_change_protein_id = str(x[2])+str(x[4])
            if catalyst_id not in complex_added_list:
                file_str += get_str_fr_complex(catalyst_id, catalyst_name, x[1])
                complex_added_list.append(catalyst_id)
            if state_change_protein_id not in complex_added_list:
                file_str += get_str_fr_complex(state_change_protein_id, state_change_protein_name, x[3])
                complex_added_list.append(state_change_protein_id)
            if new_state_change_protein_id not in complex_added_list:
                file_str += get_str_fr_complex(new_state_change_protein_id, state_change_protein_name, x[4])
                complex_added_list.append(new_state_change_protein_id)
        X_s = None
    if X_c is not None:
        n_c = X_c.shape[0]
        for curr_idx in range(n_c):
            x = X_c[curr_idx]
            catalyst_name = protein_names[x[0]]
            catalyst_id = str(x[0])+str(x[1])
            protein_1_name = protein_names[x[2]]
            protein_1_id = str(x[2])+str(x[3])
            protein_2_name = protein_names[x[4]]
            protein_2_id = str(x[4])+str(x[5])
            new_complex_name = protein_names[x[6]]
            new_complex_id = str(x[6])+str(x[7])
            if catalyst_id not in complex_added_list:
                file_str += get_str_fr_complex(catalyst_id, catalyst_name, x[1])
                complex_added_list.append(catalyst_id)
            if protein_1_id not in complex_added_list:
                file_str += get_str_fr_complex(protein_1_id, protein_1_name, x[3])
                complex_added_list.append(protein_1_id)
            if protein_2_id not in complex_added_list:
                file_str += get_str_fr_complex(protein_2_id, protein_2_name, x[5])
                complex_added_list.append(protein_2_id)
            if new_complex_id not in complex_added_list:
                file_str += get_str_fr_complex(new_complex_id, new_complex_name, x[7], [protein_1_id, protein_2_id])
                complex_added_list.append(new_complex_id)
    f = open(cap.absolute_path+file_path, 'a')
    f.write(file_str)
    f.close()


def get_catalysis_str_owl(catalyst_id, interaction_id, is_activation):
    if catalyst_id is None or interaction_id is None:
        raise AssertionError
    catalysis_str = ''
    catalysis_str += '<bp:Catalysis rdf:ID="{}">\n'.format(catalyst_id+'_'+interaction_id)
    catalysis_str += '<bp:dataSource rdf:resource="#pid" />\n'
    catalysis_str += '<bp:catalysisDirection rdf:datatype = "http://www.w3.org/2001/XMLSchema#string">LEFT_TO_RIGHT</bp:catalysisDirection>\n'
    catalysis_str += '<bp:controlled rdf:resource="#{}" />\n'.format(interaction_id)
    catalysis_str += '<bp:controlType rdf:datatype = "http://www.w3.org/2001/XMLSchema#string">{}</bp:controlType>\n'.format(('ACTIVATION' if is_activation else 'INHIBITION'))
    catalysis_str += '<bp:controller rdf:resource="#{}" />\n'.format(catalyst_id)
    catalysis_str += '</bp:Catalysis>\n'
    return catalysis_str


def write_protein_complex_formations_owl(X, protein_names, state_names, file_path):
    def get_str_fr_complex_form(x):
        complex_form_str = ''
        complex_form_str += '<bp:ComplexAssembly rdf:ID="{}">\n'
        id = 'CA'+str(x[0])+str(x[1])+str(x[2])+str(x[3])+str(x[4])+str(x[5])+str(x[6])+str(x[7])
        complex_form_str = complex_form_str.format(id)
        complex_form_str += '<bp:left rdf:resource="#{}" />\n'.format(str(x[2])+str(x[3]))
        complex_form_str += '<bp:left rdf:resource="#{}" />\n'.format(str(x[4])+str(x[5]))
        complex_form_str += '<bp:right rdf:resource="#{}" />\n'.format(str(x[6])+str(x[7]))
        complex_form_str += '</bp:ComplexAssembly>\n'
        if x[0] is not None and protein_names[x[0]] != NULL:
            if inhibit in list(get_state_string_vector_from(get_state_vector_from(x[7], k), state_names)):
                is_activate = False
            else:
                is_activate = True
            complex_form_str += get_catalysis_str_owl(str(x[0])+str(x[1]), id, is_activate)
        return complex_form_str

    if X is not None:
        #X is an array of protein state changes
        n_x = X.shape[0]
        k = state_names.size
        file_str = ''
        for curr_idx in np.arange(n_x):
            x = X[curr_idx]
            file_str += get_str_fr_complex_form(x)
        f = open(cap.absolute_path+file_path, 'a')
        f.write(file_str)
        f.close()


def write_protein_state_change_owl(X, protein_names, state_names, file_path):
    def get_str_fr_state_change(x):
        state_change_str = ''
        if inhibit in list(get_state_string_vector_from(get_state_vector_from(x[4], k), state_names)):
            #this is just inhibition of a protein
            state_change_str += '<bp:BiochemicalReaction rdf:ID="{}">\n'
            id = 'BR'+str(x[0])+str(x[1])+str(x[2])+str(x[3])
            state_change_str = state_change_str.format(id)
            state_change_str += '<bp:right rdf:resource="#{}" />\n'.format(str(x[2])+str(x[3]))
            state_change_str += '</bp:BiochemicalReaction>\n'
            if x[0] is not None and protein_names[x[0]] != NULL:
                state_change_str += get_catalysis_str_owl(str(x[0])+str(x[1]), id, False)
        else:
            state_change_str += '<bp:BiochemicalReaction rdf:ID="{}">\n'
            id = 'BR'+str(x[0])+str(x[1])+str(x[2])+str(x[3])+str(x[4])
            state_change_str = state_change_str.format(id)
            state_change_str += '<bp:left rdf:resource="#{}" />\n'.format(str(x[2])+str(x[3]))
            state_change_str += '<bp:right rdf:resource="#{}" />\n'.format(str(x[2])+str(x[4]))
            state_change_str += '</bp:BiochemicalReaction>\n'
            if x[0] is not None and protein_names[x[0]] != NULL:
                state_change_str += get_catalysis_str_owl(str(x[0])+str(x[1]), id, True)
        return state_change_str

    if X is not None:
        #X is an array of protein state changes
        n_x = X.shape[0]
        k = state_names.size
        file_str = ''
        for curr_idx in np.arange(n_x):
            x = X[curr_idx]
            file_str += get_str_fr_state_change(x)
        f = open(cap.absolute_path+file_path, 'a')
        f.write(file_str)
        f.close()


def parse_entities_info(comma_sep_file_path):
    protein_identifier = {}
    protein_family = {}
    with open(cap.absolute_path+comma_sep_file_path, 'r') as f:
        for curr_line in f:
            if debug:
                print curr_line
            curr_line = curr_line.replace('\t', '').replace('\r', '').replace('\n', '').strip()
            if curr_line:
                info_elem = curr_line.split(',')
                if debug:
                    print 'info_elem:', info_elem
                curr_protein = info_elem[0].strip()
                curr_identifier = info_elem[1].strip()
                if curr_identifier:
                    protein_identifier[curr_protein] = curr_identifier
                curr_protein_family = info_elem[2].strip()
                if curr_protein_family:
                    protein_family[curr_protein] = curr_protein_family
    return protein_identifier, protein_family
