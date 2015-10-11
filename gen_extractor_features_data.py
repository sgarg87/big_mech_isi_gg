import extract_from_amr_dot as ead
from constants_amrs_fr_domain_catalyst_paths import *
from config import *
import pickle as p
import subprocess as sp
import os
import pydot as pd
import re
from config_train_extractor import *
import constants_absolute_path as cap
import config_hpcc as ch
from config_console_output import *
import copy
import config_gen_files as cgf

is_dot = False

if not is_joint:
    const_global_train_file_name = './concept_domain_catalyst_train_data'
    const_global_test_file_name = './concept_domain_catalyst_test_data'
else:
    # joint case
    const_global_train_file_name = './concept_domain_catalyst_joint_train_data'
    const_global_test_file_name = './concept_domain_catalyst_joint_test_data'
#
const_global_protein_state_train_file_name = './protein_concept_state_train_data'
const_global_protein_state_test_file_name = './protein_concept_state_test_data'
#
const_global_joint_concept_domain_catalyst_model_data = './concept_domain_catalyst_joint_model_data'
const_global_protein_state_model_data = './protein_concept_state_model_data'
#
const_global_joint_concept_domain_catalyst_model_data_nonsynthetic_edges = './concept_domain_catalyst_joint_model_data_nonsynthetic_edges'
#
const_complex_ext = '_complex'
const_backup_ext = '_backup'
const_paths_map = 'paths_map'
const_interaction_tuples_map = 'interaction_tuples_map'
const_protein_state_tuples_map = 'protein_state_tuples_map'
const_sentences_map = 'sentences_map'
const_catalyst_labels_map = 'catalyst_labels_map'
const_domain_labels_map = 'domain_labels_map'
const_state_label_map = 'state_label_map'
const_joint_labels_map = 'joint_labels_map'
const_category_labels_map = 'category_labels_map'


def get_const_global_file_name(is_train):
    if is_train:
        return const_global_train_file_name
    else:
        return const_global_test_file_name


def get_const_global_protein_state_file_name(is_train):
    if is_train:
        return const_global_protein_state_train_file_name
    else:
        return const_global_protein_state_test_file_name


def get_amrs(amr_dot_file, start_amr, end_amr):
    amr_dot_files = []
    for i in range(start_amr, end_amr+1):
        if is_dot:
            curr_amr_dot_file = amr_dot_file + '.' + str(i) + '.dot'
        else:
            curr_amr_dot_file = amr_dot_file + str(i) + '.dot'
        if not os.path.exists(cap.absolute_path+curr_amr_dot_file+'.pdf'):
            text = re.sub('<[^>]*>', "", open(cap.absolute_path+curr_amr_dot_file).read())
            with open(cap.absolute_path+curr_amr_dot_file, "w") as f:
                f.write(text)
            g = pd.graph_from_dot_file(cap.absolute_path+curr_amr_dot_file)
            if not ch.is_hpcc and cgf.is_pdf_gen:
                g.write_pdf(cap.absolute_path+curr_amr_dot_file+'.pdf')
            g = None
        if debug:
            print 'curr_amr_dot_file:', curr_amr_dot_file
        amr_dot_files.append(curr_amr_dot_file)
    return amr_dot_files


def get_default_amrs():
    amr_dot_files = []
    for curr_amr in default_amrs:
        amr_dot_files += get_amrs(curr_amr['path'], curr_amr['idx'][0], curr_amr['idx'][1])
    return amr_dot_files


def gen_concept_domain_catalyst_data_features(amr_dot_file=None, start_amr=None, end_amr=None, protein_name_idlist_map=None,
                                              kernel_non_joint_trained_model_obj=None, proteins_filter_list=None):
    paths_map = {}
    interaction_tuples_map = {}
    sentences_map = {}
    if amr_dot_file is None:
        amr_dot_files = get_default_amrs()
    else:
        if start_amr is not None and end_amr is not None:
            amr_dot_files = get_amrs(amr_dot_file, start_amr, end_amr)
        else:
            amr_dot_files = [amr_dot_file]
    for curr_amr_dot_file in amr_dot_files:
        print 'working on ', curr_amr_dot_file
        curr_paths_map, curr_interaction_tuples_map, curr_sentences_map =\
            ead.gen_catalyst_domain_paths_frm_amr_concept_nd_wr_dot_pdf(
                curr_amr_dot_file, is_joint_path=is_joint, protein_name_idlist_map=protein_name_idlist_map, kernel_non_joint_trained_model_obj=kernel_non_joint_trained_model_obj, proteins_filter_list=proteins_filter_list)
        paths_map.update(curr_paths_map)
        interaction_tuples_map.update(curr_interaction_tuples_map)
        sentences_map.update(curr_sentences_map)
    assert(len(paths_map.values()) == len(interaction_tuples_map.values()) == len(sentences_map.values()))
    print 'Number of paths: ', len(paths_map.values())
    # dumping backup
    # temp_concept_domain_catalyst_data_features_maps = {}
    # temp_concept_domain_catalyst_data_features_maps['paths_map'] = paths_map
    # temp_concept_domain_catalyst_data_features_maps['interaction_tuples_map'] = interaction_tuples_map
    # temp_concept_domain_catalyst_data_features_maps['sentences_map'] = sentences_map
    # with open('./temp_concept_domain_catalyst_data_features_maps.pickle', 'wb') as f_t:
    #     p.dump(temp_concept_domain_catalyst_data_features_maps, f_t)
    #
    return paths_map, interaction_tuples_map, sentences_map


def gen_protein_state_data_features(amr_dot_file=None, start_amr=None, end_amr=None):
    paths_map = {}
    protein_state_tuples_map = {}
    sentences_map = {}
    if amr_dot_file is None:
        amr_dot_files = get_default_amrs()
    else:
        if start_amr is not None and end_amr is not None:
            amr_dot_files = get_amrs(amr_dot_file, start_amr, end_amr)
        else:
            amr_dot_files = [amr_dot_file]
    for curr_amr_dot_file in amr_dot_files:
        print 'working on ', curr_amr_dot_file
        curr_paths_map, curr_tuples_map, curr_sentences_map = ead.gen_concept_state_paths_frm_amr_protein_nd_wr_dot_pdf(curr_amr_dot_file)
        paths_map.update(curr_paths_map)
        protein_state_tuples_map.update(curr_tuples_map)
        sentences_map.update(curr_sentences_map)
    assert(len(paths_map.values()) == len(protein_state_tuples_map.values()) == len(sentences_map.values()))
    print 'Number of paths: ', len(paths_map.values())
    return paths_map, protein_state_tuples_map, sentences_map


def input_concept_domain_catalyst_data_labels(paths_map, is_train=True):
    def input_catalyst_label():
        try:
            curr_catalyst_label = raw_input('Enter 0 if not a catalyst else 1: ')
            curr_catalyst_label = bool(int(curr_catalyst_label))
            print '***labeling input successful: ', curr_catalyst_label
        except ValueError:
            return input_catalyst_label()
        return curr_catalyst_label

    def input_domain_label():
        try:
            curr_domain_label = raw_input('Enter 0 if not a domain else 1: ')
            curr_domain_label = bool(int(curr_domain_label))
            print '+++labeling input successful: ', curr_domain_label
        except ValueError:
            return input_domain_label()
        return curr_domain_label

    def input_state_label():
        try:
            curr_state_label = raw_input('Enter 0 if not a state else 1: ')
            curr_state_label = bool(int(curr_state_label))
            print '+++labeling input successful: ', curr_state_label
        except ValueError:
            return input_state_label()
        return curr_state_label

    catalyst_labels_map = {}
    domain_labels_map = {}
    state_label_map = {}
    path_keys = paths_map.keys()
    path_keys.sort()
    for path_key in path_keys:
        print 'Path at ', path_key
        pdf_prg_id = sp.call(['open', cap.absolute_path+path_key+'.pdf'])
        curr_catalyst_label = input_catalyst_label()
        catalyst_labels_map[path_key] = curr_catalyst_label
        curr_domain_label = input_domain_label()
        domain_labels_map[path_key] = curr_domain_label
        curr_state_label = input_state_label()
        state_label_map[path_key] = curr_state_label
        print 'dumping backup ...'
        dump_pickle_data(paths_map, None, catalyst_labels_map, domain_labels_map, state_label_map, is_backup=True, is_train=is_train)
        os.kill(pdf_prg_id, 0)
    return catalyst_labels_map, domain_labels_map, state_label_map


def input_concept_domain_catalyst_data_labels_joint(paths_map, joint_tuples_map=None, is_train=True):
    def input_joint_label():
        try:
            curr_joint_label = raw_input('Enter invalid-0, valid-1, valid-but-swap-roles-2: ')
            curr_joint_label = int(curr_joint_label)
            if curr_joint_label < 0 or curr_joint_label > 2:
                raise ValueError
            print '***labeling input successful: ', curr_joint_label
        except ValueError:
            return input_joint_label()
        return curr_joint_label

    labels_map = {}
    labels_distant_supervision_map = {}
    path_keys = paths_map.keys()
    path_keys.sort()
    count = 0
    for path_key in path_keys:
        print 'Path at ', path_key
        # dot_file_path_common = path_key[:path_key.index('.dot')+4]
        # print 'dot_file_path_common', dot_file_path_common
        curr_tuple = joint_tuples_map[path_key]
        assert (len(curr_tuple) in [3, 4])
        if len(curr_tuple) == 3:
            curr_distant_supervision_key = (curr_tuple[0].get_name_formatted(), curr_tuple[1].get_name_formatted() if curr_tuple[1] is not None else None, curr_tuple[2].get_name_formatted())
        elif len(curr_tuple) == 4:
            curr_distant_supervision_key = (curr_tuple[0].get_name_formatted(), curr_tuple[1].get_name_formatted() if curr_tuple[1] is not None else None, curr_tuple[2].get_name_formatted(), curr_tuple[3].get_name_formatted())
        else:
            raise AssertionError
        print '...........................................................................................'
        print 'curr_distant_supervision_key', curr_distant_supervision_key
        print 'labels_distant_supervision_map.keys()', labels_distant_supervision_map.keys()
        #
        if curr_distant_supervision_key in labels_distant_supervision_map:
            labels_map[path_key] = labels_distant_supervision_map[curr_distant_supervision_key]
        else:
            pdf_prg_id = sp.call(['open', cap.absolute_path+path_key+'.pdf'])
            curr_label = input_joint_label()
            labels_map[path_key] = curr_label
            labels_distant_supervision_map[curr_distant_supervision_key] = curr_label
            if len(curr_distant_supervision_key) == 4:
                curr_distant_supervision_key_swap_bind = list(copy.copy(curr_distant_supervision_key))
                curr_distant_supervision_key_swap_bind[2] = curr_distant_supervision_key[3]
                curr_distant_supervision_key_swap_bind[3] = curr_distant_supervision_key[2]
                curr_distant_supervision_key_swap_bind = tuple(curr_distant_supervision_key_swap_bind)
                if curr_distant_supervision_key_swap_bind not in labels_distant_supervision_map:
                    labels_distant_supervision_map[curr_distant_supervision_key_swap_bind] = curr_label
                if curr_label in [0, 1]:
                    if curr_label == 0:
                        curr_label_swap = 0
                    elif curr_label == 1:
                        curr_label_swap = 2
                    curr_distant_supervision_key_swap_pc_1 = list(copy.copy(curr_distant_supervision_key))
                    curr_distant_supervision_key_swap_pc_1[1] = curr_distant_supervision_key[2]
                    curr_distant_supervision_key_swap_pc_1[2] = curr_distant_supervision_key[1]
                    curr_distant_supervision_key_swap_pc_1 = tuple(curr_distant_supervision_key_swap_pc_1)
                    if curr_distant_supervision_key_swap_pc_1 not in labels_distant_supervision_map:
                        labels_distant_supervision_map[curr_distant_supervision_key_swap_pc_1] = curr_label_swap
                    curr_distant_supervision_key_swap_pc_1 = list(curr_distant_supervision_key_swap_pc_1)
                    temp = curr_distant_supervision_key_swap_pc_1[2]
                    curr_distant_supervision_key_swap_pc_1[2] = curr_distant_supervision_key_swap_pc_1[3]
                    curr_distant_supervision_key_swap_pc_1[3] = temp
                    curr_distant_supervision_key_swap_pc_1 = tuple(curr_distant_supervision_key_swap_pc_1)
                    if curr_distant_supervision_key_swap_pc_1 not in labels_distant_supervision_map:
                        labels_distant_supervision_map[curr_distant_supervision_key_swap_pc_1] = curr_label_swap
                    #
                    curr_distant_supervision_key_swap_pc_2 = list(copy.copy(curr_distant_supervision_key))
                    curr_distant_supervision_key_swap_pc_2[1] = curr_distant_supervision_key[3]
                    curr_distant_supervision_key_swap_pc_2[3] = curr_distant_supervision_key[1]
                    curr_distant_supervision_key_swap_pc_2 = tuple(curr_distant_supervision_key_swap_pc_2)
                    if curr_distant_supervision_key_swap_pc_2 not in labels_distant_supervision_map:
                        labels_distant_supervision_map[curr_distant_supervision_key_swap_pc_2] = curr_label_swap
                    curr_distant_supervision_key_swap_pc_2 = list(curr_distant_supervision_key_swap_pc_2)
                    temp = curr_distant_supervision_key_swap_pc_2[2]
                    curr_distant_supervision_key_swap_pc_2[2] = curr_distant_supervision_key_swap_pc_2[3]
                    curr_distant_supervision_key_swap_pc_2[3] = temp
                    curr_distant_supervision_key_swap_pc_2 = tuple(curr_distant_supervision_key_swap_pc_2)
                    if curr_distant_supervision_key_swap_pc_2 not in labels_distant_supervision_map:
                        labels_distant_supervision_map[curr_distant_supervision_key_swap_pc_2] = curr_label_swap
            elif (len(curr_distant_supervision_key) == 3) and (curr_distant_supervision_key[1] is not None):
                curr_distant_supervision_key_swap_catalyst_protein = list(copy.copy(curr_distant_supervision_key))
                curr_distant_supervision_key_swap_catalyst_protein[1] = curr_distant_supervision_key[2]
                curr_distant_supervision_key_swap_catalyst_protein[2] = curr_distant_supervision_key[1]
                curr_distant_supervision_key_swap_catalyst_protein = tuple(curr_distant_supervision_key_swap_catalyst_protein)
                if curr_distant_supervision_key_swap_catalyst_protein not in labels_distant_supervision_map:
                    if curr_label == 1:
                        curr_label_swap = 2
                    elif curr_label == 2:
                        curr_label_swap = 1
                    elif curr_label == 0:
                        curr_label_swap = 0
                    labels_distant_supervision_map[curr_distant_supervision_key_swap_catalyst_protein] = curr_label_swap
            os.kill(pdf_prg_id, 0)
        count += 1
        print 'No. of annotated labels so far is ', count
    print 'paths_map.keys()', paths_map.keys()
    print 'labels_map.keys()', labels_map.keys()
    print 'labels_map', labels_map
    return labels_map


def input_protein_concept_state_data_labels(paths_map, protein_state_tuples_map, is_train=True):
    def input_joint_label():
        try:
            curr_joint_label = raw_input('Enter invalid-0, valid-1, valid-but-swap-roles-2: ')
            curr_joint_label = int(curr_joint_label)
            if curr_joint_label < 0 or curr_joint_label > 2:
                raise ValueError
            print '***labeling input successful: ', curr_joint_label
        except ValueError:
            return input_joint_label()
        return curr_joint_label

    labels_map = {}
    path_keys = paths_map.keys()
    path_keys.sort()
    count = 0
    labels_distant_supervision_map = {}
    for path_key in path_keys:
        print 'Path at ', path_key
        #
        # localized distant supervision
        # for all features with same tuple in an amr, label should be same
        # this can reduce labeling effort significantly
        dot_file_path_common = path_key[:path_key.index('.dot')+4]
        curr_tuple = protein_state_tuples_map[path_key]
        assert (len(curr_tuple) == 3)
        curr_distant_supervision_key = (dot_file_path_common, curr_tuple[0].get_name_formatted(), curr_tuple[1].get_name_formatted(), curr_tuple[2].get_name_formatted())
        if curr_distant_supervision_key in labels_distant_supervision_map:
            labels_map[path_key] = labels_distant_supervision_map[curr_distant_supervision_key]
        else:
            pdf_prg_id = sp.call(['open', cap.absolute_path+path_key+'.pdf'])
            curr_label = input_joint_label()
            labels_map[path_key] = curr_label
            labels_distant_supervision_map[curr_distant_supervision_key] = curr_label
            # print 'dumping backup ...'
            # dump_pickle_data_joint(paths_map, None, labels_map, is_backup=True, is_train=is_train)
            os.kill(pdf_prg_id, 0)
        count += 1
        print 'No. of annotated labels (including distant supervised) so far is ', count
    print 'labels_distant_supervision_map is ', labels_distant_supervision_map
    print 'labels_map is ', labels_map
    return labels_map


def dump_pickle_data(paths_map, interaction_tuples_map, sentences_map, catalyst_labels_map, domain_labels_map, state_label_map, is_backup=False, old_data=None, is_train=True):
    #if old data is not None, merge new data with old data while over riding the old data in case of conflict
    if old_data is None:
        data = {}
        data[const_paths_map] = paths_map
        data[const_interaction_tuples_map] = interaction_tuples_map
        data[const_sentences_map] = sentences_map
        data[const_catalyst_labels_map] = catalyst_labels_map
        data[const_domain_labels_map] = domain_labels_map
        data[const_state_label_map] = state_label_map
    else:
        old_data[const_paths_map].update(paths_map)
        #
        if const_interaction_tuples_map in old_data:
            old_data[const_interaction_tuples_map].update(interaction_tuples_map)
        else:
            old_data[const_interaction_tuples_map] = interaction_tuples_map
        #
        if const_sentences_map in old_data:
            old_data[const_sentences_map].update(sentences_map)
        else:
            old_data[const_sentences_map] = sentences_map
        #
        old_data[const_catalyst_labels_map].update(catalyst_labels_map)
        old_data[const_domain_labels_map].update(domain_labels_map)
        old_data[const_state_label_map].update(state_label_map)
        data = old_data
    file_name = get_const_global_file_name(is_train)
    # if is_complex:
    #     file_name += const_complex_ext
    if is_backup:
        file_name += const_backup_ext
    file_name += '.pickle'
    print 'dumping the data feature and labels into the file ', file_name
    with open(cap.absolute_path+file_name, 'wb') as h:
        p.dump(data, h)


def dump_pickle_data_joint(paths_map, interaction_tuples_map, sentences_map, labels_map, is_backup=False, old_data=None, is_train=True):
    #if old data is not None, merge new data with old data while over riding the old data in case of conflict
    if old_data is None:
        data = {}
        data[const_paths_map] = paths_map
        data[const_interaction_tuples_map] = interaction_tuples_map
        data[const_sentences_map] = sentences_map
        data[const_joint_labels_map] = labels_map
    else:
        old_data[const_paths_map].update(paths_map)
        if const_interaction_tuples_map in old_data:
            old_data[const_interaction_tuples_map].update(interaction_tuples_map)
        else:
            old_data[const_interaction_tuples_map] = interaction_tuples_map
        if const_sentences_map in old_data:
            old_data[const_sentences_map].update(sentences_map)
        else:
            old_data[const_sentences_map] = sentences_map
        old_data[const_joint_labels_map].update(labels_map)
        data = old_data
    file_name = get_const_global_file_name(is_train)
    if is_backup:
        file_name += const_backup_ext
    file_name += '.pickle'
    print 'dumping the data feature and labels into the file ', file_name
    with open(cap.absolute_path+file_name, 'wb') as h:
        p.dump(data, h)


def dump_pickle_data_joint_model(data, is_extend=False, is_synthetic=False):
    if is_extend:
        raise NotImplementedError
    if is_synthetic:
        file_name = const_global_joint_concept_domain_catalyst_model_data
    else:
        file_name = const_global_joint_concept_domain_catalyst_model_data_nonsynthetic_edges
    file_name += '.pickle'
    print 'dumping the data feature and labels into the file ', file_name
    with open(cap.absolute_path+file_name, 'wb') as h:
        p.dump(data, h)


def dump_pickle_data_protein_state(paths_map, protein_state_tuples_map, sentences_map, labels_map, is_backup=False, old_data=None, is_train=True):
    #if old data is not None, merge new data with old data while over riding the old data in case of conflict
    if old_data is None:
        data = {}
        data[const_paths_map] = paths_map
        data[const_protein_state_tuples_map] = protein_state_tuples_map
        data[const_sentences_map] = sentences_map
        data[const_joint_labels_map] = labels_map
    else:
        old_data[const_paths_map].update(paths_map)
        old_data[const_protein_state_tuples_map].update(protein_state_tuples_map)
        old_data[const_sentences_map].update(sentences_map)
        old_data[const_joint_labels_map].update(labels_map)
        data = old_data
    file_name = get_const_global_protein_state_file_name(is_train)
    if is_backup:
        file_name += const_backup_ext
    file_name += '.pickle'
    print 'dumping the data feature and labels into the file ', file_name
    with open(cap.absolute_path+file_name, 'wb') as h:
        p.dump(data, h)


def dump_pickle_data_protein_state_model(data, is_extend=False):
    if is_extend:
        raise NotImplementedError
    file_name = const_global_protein_state_model_data
    file_name += '.pickle'
    print 'dumping the data feature and labels into the file ', file_name
    with open(cap.absolute_path+file_name, 'wb') as h:
        p.dump(data, h)


def merge_state_change_nd_complex_form_data(state_change_data, complex_data):
    if state_change_data is None and complex_data is None:
        raise AssertionError
    elif state_change_data is None:
        return complex_data
    elif complex_data is None:
        return state_change_data
    else:
        #merge the two data sets
        data = {}
        #paths and corresponding path nodes list
        data[const_paths_map] = {}
        data[const_paths_map].update(state_change_data[const_paths_map])
        data[const_paths_map].update(complex_data[const_paths_map])
        #
        data[const_interaction_tuples_map] = {}
        if const_interaction_tuples_map in state_change_data:
            data[const_interaction_tuples_map].update(state_change_data[const_interaction_tuples_map])
        if const_interaction_tuples_map in complex_data:
            data[const_interaction_tuples_map].update(complex_data[const_interaction_tuples_map])
        #
        #catalyst labels for paths
        data[const_catalyst_labels_map] = {}
        data[const_catalyst_labels_map].update(state_change_data[const_catalyst_labels_map])
        data[const_catalyst_labels_map].update(complex_data[const_catalyst_labels_map])
        #domain labels for paths
        data[const_domain_labels_map] = {}
        data[const_domain_labels_map].update(state_change_data[const_domain_labels_map])
        data[const_domain_labels_map].update(complex_data[const_domain_labels_map])
        #state labels for paths
        data[const_state_label_map] = {}
        data[const_state_label_map].update(state_change_data[const_state_label_map])
        data[const_state_label_map].update(complex_data[const_state_label_map])
        return data


def load_pickled_merged_data(is_train=True):
    state_change_data, complex_data = load_pickled_data(is_train)
    data = merge_state_change_nd_complex_form_data(state_change_data, complex_data)
    if const_sentences_map not in data:
        data[const_sentences_map] = {}
    return data


def load_pickled_data(is_train=True):
    def extend_file_name(file_name):
        file_name += '.pickle'
        return file_name

    #state change data
    try:
        file_name = extend_file_name(get_const_global_file_name(is_train))
        with open(cap.absolute_path+file_name, 'r') as h:
            state_change_data = p.load(h)
    except:
        state_change_data = None
    #complex data
    try:
        file_name = extend_file_name(get_const_global_file_name(is_train) + const_complex_ext)
        with open(cap.absolute_path+file_name, 'r') as h:
            complex_data = p.load(h)
    except:
        complex_data = None
    return state_change_data, complex_data


def load_pickled_joint_data_model(is_synthetic=False):
    if is_synthetic:
        file_name = const_global_joint_concept_domain_catalyst_model_data + '.pickle'
    else:
        file_name = const_global_joint_concept_domain_catalyst_model_data_nonsynthetic_edges + '.pickle'
    with open(cap.absolute_path+file_name, 'r') as h:
        data_model = p.load(h)
    return data_model


def load_pickled_protein_state_data(is_train=True):
    def extend_file_name(file_name):
        file_name += '.pickle'
        return file_name

    file_name = extend_file_name(get_const_global_protein_state_file_name(is_train))
    with open(cap.absolute_path+file_name, 'r') as h:
        data = p.load(h)
    return data


def load_pickled_protein_state_data_model():
    file_name = const_global_protein_state_model_data+'.pickle'
    with open(cap.absolute_path+file_name, 'r') as h:
        data = p.load(h)
    return data


def input_nd_dump_interaction_category_fr_concept_domain_catalyst_data_joint(is_train=True):
    raise NotImplementedError
    # todo: catalyst category (control category) and information_category (positive and negative) should be two different things)
    def dump_data():
        file_name = get_const_global_file_name(is_train)
        file_name += '.pickle'
        print 'dumping the data with annotated category labels into the file ', file_name
        with open(cap.absolute_path+file_name, 'wb') as h:
            p.dump(data, h)

    def input_category_label():
        try:
            curr_joint_label = raw_input('Enter decrease-0, increase-1, neutral-2, correlate-3: ')
            curr_joint_label = int(curr_joint_label)
            if curr_joint_label < 0 or curr_joint_label > 3:
                raise ValueError
            print '***labeling input successful: ', curr_joint_label
        except ValueError:
            return input_category_label()
        return curr_joint_label

    if is_train:
        data = load_pickled_merged_data(is_train=True)
    else:
        data = load_pickled_merged_data(is_train=False)
    graph_keys = data['paths_map'].keys()
    graph_keys.sort()
    n = len(graph_keys)
    print 'Number of graphs: ', n
    if len(data[const_joint_labels_map].values()) != n:
        raise AssertionError
    if const_category_labels_map not in data:
        data[const_category_labels_map] = {}
    label_count = 0
    for i in range(n):
        curr_graph_key = graph_keys[i]
        print 'curr_graph_key ', curr_graph_key
        if curr_graph_key in data[const_category_labels_map]:
            print 'Label already input in the past.'
        else:
            print 'interaction joint validity label is ', data[const_joint_labels_map][curr_graph_key]
            if data[const_joint_labels_map][curr_graph_key] == 0: #invalid case
                print 'Default label for invalid interaction is -1. So need for input'
                data[const_category_labels_map][curr_graph_key] = -1
            else:
                # curr_nodes_list = data[const_paths_map][curr_graph_key]
                # curr_nodes_list = ead.remove_duplicates_based_on_ids(curr_nodes_list)
                # curr_nodes_list = ead.prune_non_path_nodes_references_frm_subgraph(curr_nodes_list)
                # ead.nodes_to_dot(curr_nodes_list, curr_graph_key)
                pdf_prg_id = sp.call(['open', cap.absolute_path+curr_graph_key+'.pdf'])
                curr_label = input_category_label()
                data[const_category_labels_map][curr_graph_key] = curr_label
                os.kill(pdf_prg_id, 0)
                print 'No. of annotated category labels so far is ', i+1
                label_count += 1
                if (label_count % 20) == 0:
                    try:
                        is_dump = raw_input('Should I dump the current data ? If yes, press 1 otherwise 0.: ')
                        is_dump = bool(int(is_dump))
                    except ValueError:
                        print 'error in user response to the question, so dumping to be on safe side ...'
                        is_dump = True
                    if is_dump:
                        dump_data()
    dump_data()


if __name__ == '__main__':
    import sys
    is_gen_new_data_features_nd_label = bool(sys.argv[1])
    if is_gen_new_data_features_nd_label:
        if len(sys.argv) > 2:
            amr_dot_file = str(sys.argv[2])
            start_amr = int(sys.argv[3])
            end_amr = int(sys.argv[4])
        else:
            amr_dot_file = None
        if len(sys.argv) > 5:
            is_extend_data = bool(sys.argv[5])
        else:
            is_extend_data = True #to be on safe side, it is always better to extend on default
        if len(sys.argv) > 6:
            is_train = bool(sys.argv[6])
        else:
            is_train = True #default
        if len(sys.argv) > 7:
            is_protein_state = bool(sys.argv[7])
        else:
            is_protein_state = False
        if not is_protein_state:
            if amr_dot_file is None:
                paths_map, interaction_tuples_map, sentences_map = gen_concept_domain_catalyst_data_features()
            else:
                paths_map, interaction_tuples_map, sentences_map = gen_concept_domain_catalyst_data_features(amr_dot_file, start_amr, end_amr)
            if not is_joint:
                catalyst_labels_map, domain_labels_map, state_label_map = input_concept_domain_catalyst_data_labels(paths_map, is_train=is_train)
            else:
                joint_labels_map = input_concept_domain_catalyst_data_labels_joint(paths_map, joint_tuples_map=interaction_tuples_map, is_train=is_train)
            if is_extend_data:
                old_data = load_pickled_merged_data(is_train=is_train)
                if not is_joint:
                    dump_pickle_data(paths_map, interaction_tuples_map, sentences_map, catalyst_labels_map, domain_labels_map, state_label_map, old_data=old_data, is_train=is_train)
                else:
                    dump_pickle_data_joint(paths_map, interaction_tuples_map, sentences_map, joint_labels_map, old_data=old_data, is_train=is_train)
            else:
                if not is_joint:
                    dump_pickle_data(paths_map, interaction_tuples_map, sentences_map, catalyst_labels_map, domain_labels_map, state_label_map, is_train=is_train)
                else:
                    dump_pickle_data_joint(paths_map, interaction_tuples_map, sentences_map, joint_labels_map, is_train=is_train)
        else:
            if amr_dot_file is None:
                paths_map, tuples_map, sentences_map = gen_protein_state_data_features()
            else:
                paths_map, tuples_map, sentences_map = gen_protein_state_data_features(amr_dot_file, start_amr, end_amr)
            labels_map = input_protein_concept_state_data_labels(paths_map, tuples_map, is_train=is_train)
            if is_extend_data:
                old_data = load_pickled_protein_state_data(is_train=is_train)
                dump_pickle_data_protein_state(paths_map, tuples_map, sentences_map, labels_map, old_data=old_data, is_train=is_train)
            else:
                dump_pickle_data_protein_state(paths_map, tuples_map, sentences_map, labels_map, is_train=is_train)
    else:
        raise NotImplementedError
        if len(sys.argv) > 3:
            raise AssertionError
        is_train = bool(sys.argv[2])
        input_nd_dump_interaction_category_fr_concept_domain_catalyst_data_joint(is_train=is_train)

