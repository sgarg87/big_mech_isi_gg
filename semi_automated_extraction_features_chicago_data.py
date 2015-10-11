import glob
import gen_extractor_features_data as gtd
import constants_absolute_path as cap
import pickle as p
import json
import eval_divergence_frm_kernel as edk
import difflib as dl
import extract_from_amr_dot as ead
import pydot as pd
import parallel_computing as pk
import numpy as np
import copy
import subprocess as sp
import os


path = '../../chicago/'

# activate
# transactivate
# regulate
# translocate
# induce
# synthesize
# stimulate
# bind
# express
# signal
# phosphorylate
# decrease
# associate
# interact
# immunoprecipitate
# assemble
# transcribe
# inhibit
# mutate
# increase
# ubiquitinate
# release
# repress
# complex
# generate
# potentiate
# downregulate
# block
# modulate
# form
# elevate
# coimmunoprecipitate
# synergy
# suppress
# disrupt
# secrete
# translate
# iodinate
# coexpress
# acetylate
# produce
# recruit
# dephosphorylate
# glycosylate
# deaminate
# overexpress
# heterodimer
# degrade
# play
# polymerize
# localize
# dimerize
# synergistic
# sensitize
# dissociate
# co-express-00
# sumoylate
# colocalize
# hydrolyze
# upregulate
# disassemble
# deacetylate
# exert
# separate
# coprecipitate
# methylate
# myristoylate
# deactivate
# relocalize
# hyperphosphorylate
# bond
# homologue
# replicate
# hypophosphorylate
# synergism
# potent
# co-immunoprecipitate-00
# repressor
# synergize
# ribosylate
# heterodimerize
# farnesylate
# demethylate
# disassociate
# synergistically
#


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


def test_all_graphs():
    #
    f = open(cap.absolute_path+'./list_of_chicago_dot_files_not_processed.txt', 'w')
    #
    dot_files_list = get_all_files()
    for curr_file in dot_files_list:
        try:
            print curr_file
            suffix = '.dot'
            assert curr_file.endswith(suffix)
            curr_file_sub = curr_file[:-len(suffix)]
            #
            amr = pd.graph_from_dot_file(cap.absolute_path+curr_file)
        except BaseException as e:
            print 'failed to process {}'.format(curr_file)
            print e
            f.write(curr_file)
    #
    f.close()


def get_all_files():
    assert len(glob.glob(cap.absolute_path+path+'*joint*')) == 0
    #
    dot_files_list = glob.glob(cap.absolute_path+path+'*.dot')
    return dot_files_list


def generate_subgraphs_nd_dump(num_cores, curr_core):
    gold_proteins_list_map = load_proteins_list_map()
    #
    paths_map = {}
    interaction_tuples_map = {}
    sentences_map = {}
    labels_map = {}
    #
    f = open(cap.absolute_path+'./list_of_chicago_dot_files_not_processed_{}_{}.txt'.format(num_cores, curr_core), 'w')
    #
    dot_files_list = get_all_files()
    n = len(dot_files_list)
    #
    idx_range_parallel = pk.uniform_distribute_tasks_across_cores(n, num_cores)
    dot_files_list = np.array(dot_files_list)
    dot_files_list_curr_core = dot_files_list[idx_range_parallel[curr_core]]
    dot_files_list = None
    #
    count = 0
    #
    for curr_file in dot_files_list_curr_core:
        try:
            print 'curr_file', curr_file
            prefix = path+'stat_ID'
            suffix = '.dot'
            assert curr_file.startswith(cap.absolute_path+prefix)
            curr_file_sub = curr_file[len(cap.absolute_path+prefix):]
            print 'curr_file_sub', curr_file_sub
            assert curr_file_sub.endswith(suffix)
            curr_file_sub = curr_file_sub[:-len(suffix)]
            print 'curr_file_sub', curr_file_sub
            #
            curr_id = 'ID'+curr_file_sub
            curr_sentence_id = int(curr_file_sub)
            print 'curr_sentence_id', curr_sentence_id
            #
            #
            start_amr = curr_sentence_id
            end_amr = curr_sentence_id
            amr_dot_file = prefix
            print 'amr_dot_file', amr_dot_file
            #
            curr_gold_proteins_list = gold_proteins_list_map[curr_id]
            #
            curr_paths_map, curr_interaction_tuples_map, curr_sentences_map =\
                gtd.gen_concept_domain_catalyst_data_features(amr_dot_file, start_amr, end_amr, proteins_filter_list=curr_gold_proteins_list)
            print curr_paths_map.keys()
            #
            paths_map.update(curr_paths_map)
            interaction_tuples_map.update(curr_interaction_tuples_map)
            sentences_map.update(curr_sentences_map)
            #
            count += 1
            if (count % 1000) == 10:
                dump_pickle_data_joint(num_cores, curr_core, paths_map, interaction_tuples_map, sentences_map, labels_map)
        except:
            print 'failed to process {}'.format(curr_file)
            f.write(curr_file+'\n')
    #
    f.close()
    labels_map = {}
    dump_pickle_data_joint(num_cores, curr_core, paths_map, interaction_tuples_map, sentences_map, labels_map)


def dump_pickle_data_joint(num_cores, curr_core, paths_map, interaction_tuples_map, sentences_map, labels_map, is_labeled=False):
    data = {}
    data[gtd.const_paths_map] = paths_map
    data[gtd.const_interaction_tuples_map] = interaction_tuples_map
    data[gtd.const_sentences_map] = sentences_map
    data[gtd.const_joint_labels_map] = labels_map
    #
    file_name = get_file_path(num_cores, curr_core)
    print 'dumping the data feature and labels into the file ', file_name
    #
    if is_labeled:
        labeled_prefix = 'labeled_'
    else:
        labeled_prefix = ''
    #
    with open(cap.absolute_path+labeled_prefix+file_name, 'wb') as h:
        p.dump(data, h)


def load_pickled_data_joint(num_cores):
    print 'loading ...'
    paths_map = {}
    interaction_tuples_map = {}
    sentences_map = {}
    labels_map = {}
    for curr_core in range(num_cores):
        file_name = get_file_path(num_cores, curr_core)
        print file_name
        with open(cap.absolute_path+file_name, 'rb') as f:
            curr_data = p.load(f)
            paths_map.update(curr_data[gtd.const_paths_map])
            print len(paths_map)
            interaction_tuples_map.update(curr_data[gtd.const_interaction_tuples_map])
            print len(interaction_tuples_map)
            sentences_map.update(curr_data[gtd.const_sentences_map])
            print len(sentences_map)
            labels_map.update(curr_data[gtd.const_joint_labels_map])
            print len(labels_map)
    #
    data = {}
    data[gtd.const_paths_map] = paths_map
    data[gtd.const_interaction_tuples_map] = interaction_tuples_map
    data[gtd.const_sentences_map] = sentences_map
    data[gtd.const_joint_labels_map] = labels_map
    print 'done.'
    return data


def load_labeled_pickled_data_joint():
    labeled_prefix = 'labeled_'
    file_name = labeled_prefix + get_file_path(None, None)
    print 'loading from {} ...'.format(file_name)
    with open(cap.absolute_path+file_name, 'rb') as f:
        data = p.load(f)
    print 'done.'
    return data


def load_proteins_list_map():
    with open(cap.absolute_path+'../chicago_data/stats_proteins.json', 'r') as f:
        proteins_list_map = json.load(f)
    return proteins_list_map


def load_sentence_id_relations_list_map():
    with open(cap.absolute_path+'../chicago_data/stats_dataout.json', 'r') as f:
        sentence_relations_map = json.load(f)
    return sentence_relations_map


def get_passage_sentence_id_frm_joint_graph_path(joint_graph_path):
    dot_idx = joint_graph_path.index('.dot')
    joint_graph_path = joint_graph_path[:dot_idx]
    dot_idx = None
    prefix = '../../chicago/stat_ID'
    assert joint_graph_path.startswith(cap.absolute_path+prefix)
    joint_graph_path = joint_graph_path[len(prefix):]
    passage_sentence_id = int(joint_graph_path)
    return passage_sentence_id


def preprocess_extracted_interactions(data):
    paths_map = data[gtd.const_paths_map]
    assert paths_map is not None and paths_map
    #
    interaction_tuples_map = data[gtd.const_interaction_tuples_map]
    assert interaction_tuples_map is not None and interaction_tuples_map
    #
    sentences_map = data[gtd.const_sentences_map]
    assert sentences_map is not None and sentences_map
    #
    labels_map = data[gtd.const_joint_labels_map]
    assert labels_map is not None
    #
    assert not labels_map
    #
    list_of_paths = copy.copy(interaction_tuples_map.keys())
    print len(list_of_paths)
    for curr_path in list_of_paths:
        curr_interaction_tuple = interaction_tuples_map[curr_path]
        # #
        curr_interaction_str_tuple =\
            edk.get_triplet_str_tuple(curr_interaction_tuple, is_concept_mapped_to_interaction=False)
        #
        if (len(curr_interaction_tuple) == 3 and curr_interaction_tuple[1] is None) \
                or (len(curr_interaction_tuple) == 4 and curr_interaction_tuple[1] is not None):
            # print curr_interaction_str_tuple
            paths_map.pop(curr_path, None)
            interaction_tuples_map.pop(curr_path, None)
            sentences_map.pop(curr_path, None)
            continue
        elif len(curr_interaction_tuple) == 4:
            assert curr_interaction_tuple[1] is None
            #
            curr_interaction_tuple = list(curr_interaction_tuple)
            curr_interaction_tuple.remove(None)
            assert len(curr_interaction_tuple) == 3
            curr_interaction_tuple = tuple(curr_interaction_tuple)
            interaction_tuples_map[curr_path] = curr_interaction_tuple
            #
    #
    print len(paths_map)
    print len(interaction_tuples_map)
    print len(sentences_map)


def get_matched_ids_list():
    with open('./chicago_matched_ids_list.json', 'r') as f:
        matched_ids_list = json.load(f)
    return matched_ids_list


def filter_data_on_ids():
    #
    matched_ids_list = get_matched_ids_list()
    #
    data = load_labeled_pickled_data_joint()
    #
    paths_map = data[gtd.const_paths_map]
    assert paths_map is not None and paths_map
    #
    interaction_tuples_map = data[gtd.const_interaction_tuples_map]
    assert interaction_tuples_map is not None and interaction_tuples_map
    #
    sentences_map = data[gtd.const_sentences_map]
    assert sentences_map is not None and sentences_map
    #
    labels_map = data[gtd.const_joint_labels_map]
    assert labels_map is not None and labels_map
    #
    paths_list = copy.copy(interaction_tuples_map.keys())
    #
    print 'paths_map', len(paths_map)
    #
    num_pos = 0
    num_swap = 0
    num_neg = 0
    #
    interactions_type_list = []
    #
    for curr_path in paths_list:
        curr_sentence_id = get_passage_sentence_id_frm_joint_graph_path(curr_path)
        curr_id = 'ID'+str(curr_sentence_id)
        #
        if curr_id not in matched_ids_list:
            paths_map.pop(curr_path, None)
            interaction_tuples_map.pop(curr_path, None)
            sentences_map.pop(curr_path, None)
            labels_map.pop(curr_path, None)
        else:
            curr_interaction_tuple = interaction_tuples_map[curr_path]
            curr_interaction_str_tuple =\
                edk.get_triplet_str_tuple(curr_interaction_tuple, is_concept_mapped_to_interaction=False)
            curr_interaction_tuple = None
            curr_interaction_type = curr_interaction_str_tuple[0]
            if curr_interaction_type not in interactions_type_list:
                print curr_interaction_type
                interactions_type_list.append(curr_interaction_type)
            #
            curr_label = labels_map[curr_path]
            if curr_label == 1:
                num_pos += 1
            elif curr_label == 2:
                num_swap += 1
            elif curr_label == 0:
                num_neg += 1
            else:
                raise AssertionError
    #
    print 'paths_map', len(paths_map)
    #
    print '1', num_pos
    print '2', num_swap
    print '0', num_neg
    dump_pickle_data_joint(None, None, paths_map, interaction_tuples_map, sentences_map, labels_map, is_labeled=True)


def filter_proteins_list():
    def save_proteins_valid_map(proteins_valid_map):
        f = open('./chicago_proteins_valid_map.json', 'w')
        json.dump(proteins_valid_map, f, indent=4)
        f.close()

    def input_protein_label():
        try:
            curr_label = raw_input('Enter valid-1, invalid-0: ')
            curr_label = int(curr_label)
            if curr_label < 0 or curr_label > 1:
                raise ValueError
            print '***labeling input successful: ', curr_label
        except ValueError:
            return input_protein_label()
        return curr_label

    proteins_list_map = load_proteins_list_map()
    proteins_valid_map = {}
    for curr_id in proteins_list_map:
        curr_proteins_list = proteins_list_map[curr_id]
        curr_path = '../../chicago/stat_'+curr_id
        amr = pd.graph_from_dot_file(cap.absolute_path+curr_path+'.dot')
        amr.write_pdf(cap.absolute_path+curr_path+'.pdf')
        pdf_prg_id = sp.call(['open', cap.absolute_path+curr_path+'.pdf'])
        for curr_protein in curr_proteins_list:
            print '****************'
            print 'curr_protein: ', curr_protein
            if curr_protein not in proteins_list_map:
                curr_label = input_protein_label()
                print 'curr_label', curr_label
                proteins_valid_map[curr_protein] = curr_label
        #
        os.kill(pdf_prg_id, 0)
        print 'No. of annotated category labels so far is ', len(proteins_valid_map)
        save_proteins_valid_map(proteins_valid_map)
    #
    save_proteins_valid_map(proteins_valid_map)


def label_subgraphs_nd_dump(num_cores):
    data = load_pickled_data_joint(num_cores)
    preprocess_extracted_interactions(data)
    #
    sentence_relations_map = load_sentence_id_relations_list_map()
    #
    paths_map = data[gtd.const_paths_map]
    assert paths_map is not None and paths_map
    interaction_tuples_map = data[gtd.const_interaction_tuples_map]
    assert interaction_tuples_map is not None and interaction_tuples_map
    sentences_map = data[gtd.const_sentences_map]
    assert sentences_map is not None and sentences_map
    labels_map = data[gtd.const_joint_labels_map]
    assert labels_map is not None
    assert not labels_map
    #
    print len(paths_map)
    print len(interaction_tuples_map)
    print len(sentences_map)
    print len(labels_map)
    #
    num_org_positive_labels = 0
    num_org_negative_labels = 0
    positive_label_org_interactions_list = []
    negative_label_org_interactions_list = []
    #
    positive_interaction_types_list = []
    #
    for curr_id in sentence_relations_map:
        list_gold_relations = sentence_relations_map[curr_id]
        #
        for curr_relation in list_gold_relations:
            curr_relation_label = curr_relation[1]
            curr_relation_tuple = curr_relation[0]
            curr_relation_type = curr_relation_tuple[0]
            #
            if curr_relation_label == 1:
                if curr_relation_type not in positive_interaction_types_list:
                    positive_interaction_types_list.append(curr_relation_type)
                if curr_relation_tuple not in positive_label_org_interactions_list:
                    positive_label_org_interactions_list.append(curr_relation_tuple)
                    num_org_positive_labels += 1
            elif curr_relation_label == 0:
                if curr_relation_tuple not in negative_label_org_interactions_list:
                    negative_label_org_interactions_list.append(curr_relation_tuple)
                    num_org_negative_labels += 1
            else:
                raise AssertionError
    #
    with open('../chicago_data/positive_interaction_types_list.json', 'w') as f:
        json.dump(positive_interaction_types_list, f, indent=4)
    #
    num_positive_labels = 0
    num_negative_labels = 0
    positive_label_interactions_list = []
    negative_label_interactions_list = []
    #
    is_global_compare = False
    if is_global_compare:
        list_gold_relations = positive_label_org_interactions_list
    #
    symmetric_types_list = \
        [
        'bind', 'interact', 'associate', 'dissociate', 'dissociate from',
        'colocalize', 'synergy', ' synergistic interaction',
         'synergistic action', 'act synergistically', 'break',
        'form a complex', 'disassemble', 'combine', 'tie', 'attach',
        'assemble', 'form', 'couple', 'unpair', 'coexpress', 'connect',
        'pair', 'synergistic cooperation', 'work synergistically',
        'function synergistically', 'bond', 'disengage', 'synergistic association',
        'dissociate', 'dissociate from', 'copurify', 'join', 'independence',
        'synergistic integration']
    #
    for curr_path in interaction_tuples_map:
        curr_sentence = sentences_map[curr_path]
        #
        curr_interaction_tuple = interaction_tuples_map[curr_path]
        #
        curr_interaction_str_tuple =\
            edk.get_triplet_str_tuple(curr_interaction_tuple, is_concept_mapped_to_interaction=False)
        curr_interaction_tuple = None
        #
        curr_interaction_type = curr_interaction_str_tuple[0]
        #
        list_of_proteins = list(curr_interaction_str_tuple[1:])
        assert None not in list_of_proteins
        assert len(list_of_proteins) == 2
        #
        curr_sentence_id = get_passage_sentence_id_frm_joint_graph_path(curr_path)
        curr_id = 'ID'+str(curr_sentence_id)
        assert curr_id in sentence_relations_map
        if not is_global_compare:
            list_gold_relations = sentence_relations_map[curr_id]
        #
        curr_label = 0
        for curr_relation in list_gold_relations:
            if is_global_compare:
                curr_relation_proteins = curr_relation[1:]
                curr_relation_interaction_type = curr_relation[0]
                curr_relation_label = 1
                curr_relation_str_tuple = curr_relation
            else:
                curr_relation_proteins = curr_relation[0][1:]
                curr_relation_interaction_type = curr_relation[0][0]
                curr_relation_label = curr_relation[1]
                curr_relation_str_tuple = curr_relation[0]
            #
            if match_protein_name(curr_interaction_type, curr_relation_interaction_type) and curr_relation_label == 1:
                assert len(curr_relation_proteins) == 2
                if match_protein_name(list_of_proteins[0], curr_relation_proteins[0]) and match_protein_name(list_of_proteins[1], curr_relation_proteins[1]):
                    curr_label = 1
                    break
                elif match_protein_name(list_of_proteins[0], curr_relation_proteins[1]) and match_protein_name(list_of_proteins[1], curr_relation_proteins[0]):
                    if curr_relation_interaction_type in symmetric_types_list:
                        curr_label = 1
                        break
                    else:
                        curr_label = 2
        #
        labels_map[curr_path] = curr_label
        if curr_label == 1:
            if curr_interaction_str_tuple not in positive_label_interactions_list:
                positive_label_interactions_list.append(curr_interaction_str_tuple)
                num_positive_labels += 1
        else:
            if curr_interaction_str_tuple not in negative_label_interactions_list:
                negative_label_interactions_list.append(curr_interaction_str_tuple)
                # print '**************************'
                # print curr_sentence
                # print curr_interaction_str_tuple
                num_negative_labels += 1
    #
    # matched_org_positive_interactions_count = 0
    not_matched_org_positive_interactions_list = []
    matched_org_positive_interactions_list = []
    matched_ids_list = []
    for curr_id in sentence_relations_map:
        curr_gold_relations_list = sentence_relations_map[curr_id]
        for curr_org_interaction in curr_gold_relations_list:
            if curr_org_interaction[1] == 1:
                curr_org_interaction = curr_org_interaction[0]
                is_matched = False
                for curr_interaction in positive_label_interactions_list:
                    if match_protein_name(curr_org_interaction[0], curr_interaction[0]):
                        if match_protein_name(curr_org_interaction[1], curr_interaction[1]) \
                                and match_protein_name(curr_org_interaction[2], curr_interaction[2]):
                            is_matched = True
                            break
                        elif match_protein_name(curr_org_interaction[1], curr_interaction[2]) \
                                and match_protein_name(curr_org_interaction[2], curr_interaction[1]):
                            if curr_org_interaction[0] in symmetric_types_list:
                                is_matched = True
                                break
                #
                if is_matched:
                    if curr_id not in matched_ids_list:
                        matched_ids_list.append(curr_id)
                    #
                    if curr_org_interaction not in matched_org_positive_interactions_list:
                        matched_org_positive_interactions_list.append(curr_org_interaction)
                else:
                    # curr_unmatched_tuple = tuple([curr_id, curr_org_interaction])
                    # print curr_unmatched_tuple
                    if curr_org_interaction not in not_matched_org_positive_interactions_list:
                        not_matched_org_positive_interactions_list.append(curr_org_interaction)
    #
    print 'matched_ids_list', len(matched_ids_list)
    with open('./chicago_matched_ids_list.json', 'w') as f:
        json.dump(matched_ids_list, f, indent=4)
    #
    print 'matched_org_positive_interactions_list', len(matched_org_positive_interactions_list)
    with open('./chicago_matched_org_positive_interactions_list.json', 'w') as f:
        json.dump(matched_org_positive_interactions_list, f, indent=4)
    #
    print 'not_matched_org_positive_interactions_list', len(not_matched_org_positive_interactions_list)
    with open('./chicago_not_matched_org_positive_interactions_list.json', 'w') as f:
        json.dump(not_matched_org_positive_interactions_list, f, indent=4)
    #
    print 'number of positive labeled graphs: ', num_positive_labels
    print 'number of negative labeled graphs: ', num_negative_labels
    print 'total no. of labels: ', num_positive_labels+num_negative_labels
    #
    print 'original positive labels was: ', num_org_positive_labels
    print 'original negative labels was: ', num_org_negative_labels
    #
    dump_pickle_data_joint(None, None, paths_map, interaction_tuples_map, sentences_map, labels_map, is_labeled=True)


def get_file_path(num_cores, curr_core):
    if curr_core is None and num_cores is None:
        file_name = 'chicago_concept_domain_catalyst_data'
    else:
        assert curr_core is not None and num_cores is not None
        file_name = 'chicago_concept_domain_catalyst_data//num_cores_{}_curr_core_{}'.format(num_cores, curr_core)
    #
    file_name += '.pickle'
    return file_name


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        num_cores = int(sys.argv[1])
        curr_core = int(sys.argv[2])
    else:
        num_cores = 1
        curr_core = 1
    #
    generate_subgraphs_nd_dump(num_cores, curr_core)

