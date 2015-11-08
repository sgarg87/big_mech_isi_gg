import glob
import gen_extractor_features_data as gtd
import constants_absolute_path as cap
import pickle as p
import json
import eval_divergence_frm_kernel as edk
import difflib as dl


aimed_file_name = 'aimed_concept_domain_catalyst_data'
aimed_file_name += '.pickle'


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


def generate_subgraphs_nd_dump():
    path = '../../aimed/'
    #
    paths_map = {}
    interaction_tuples_map = {}
    sentences_map = {}
    #
    assert len(glob.glob(path+'*joint*')) == 0
    f = open(cap.absolute_path+'./list_of_aimed_dot_files_not_processed.txt', 'w')
    #
    for curr_file in glob.glob(cap.absolute_path+path+'*.dot'):
        try:
            print 'curr_file', curr_file
            prefix = path
            suffix = '.dot'
            assert curr_file.startswith(prefix)
            curr_file_sub = curr_file[len(prefix):]
            print 'curr_file_sub', curr_file_sub
            assert curr_file_sub.endswith(suffix)
            curr_file_sub = curr_file_sub[:-len(suffix)]
            print 'curr_file_sub', curr_file_sub
            curr_file_sub_split = curr_file_sub.split('.')
            assert len(curr_file_sub_split) == 2
            #
            curr_document = curr_file_sub_split[0]
            print 'curr_document', curr_document
            curr_sentence_id = int(curr_file_sub_split[1])
            print 'curr_sentence_id', curr_sentence_id
            #
            start_amr = curr_sentence_id
            end_amr = curr_sentence_id
            amr_dot_file = prefix+curr_document
            print 'amr_dot_file', amr_dot_file
            #
            curr_paths_map, curr_interaction_tuples_map, curr_sentences_map =\
                gtd.gen_concept_domain_catalyst_data_features(amr_dot_file, start_amr, end_amr)
            #
            paths_map.update(curr_paths_map)
            interaction_tuples_map.update(curr_interaction_tuples_map)
            sentences_map.update(curr_sentences_map)
            #
        except:
            f.write(curr_file)
    #
    f.close()
    labels_map = {}
    dump_aimed_pickle_data_joint(paths_map, interaction_tuples_map, sentences_map, labels_map)


def dump_aimed_pickle_data_joint(paths_map, interaction_tuples_map, sentences_map, labels_map, is_filtered=False, is_labeled=False):
    data = {}
    data[gtd.const_paths_map] = paths_map
    data[gtd.const_interaction_tuples_map] = interaction_tuples_map
    data[gtd.const_sentences_map] = sentences_map
    data[gtd.const_joint_labels_map] = labels_map
    print 'dumping the data feature and labels into the file ', aimed_file_name
    #
    if is_filtered:
        filtered_prefix = 'filtered_'
    else:
        filtered_prefix = ''
    #
    if is_labeled:
        labeled_prefix = 'labeled_'
    else:
        labeled_prefix = ''
    #
    with open(cap.absolute_path+filtered_prefix+labeled_prefix+aimed_file_name, 'wb') as h:
        p.dump(data, h)


def load_aimed_pickled_data_joint():
    print 'loading ...'
    with open(cap.absolute_path+aimed_file_name, 'rb') as f:
        data = p.load(f)
    print 'done.'
    return data


def load_aimed_pickled_filtered_data_joint():
    print 'loading ...'
    with open(cap.absolute_path+'filtered_'+aimed_file_name, 'rb') as f:
        data = p.load(f)
    print 'done.'
    return data


def load_aimed_pickled_filtered_labeled_data_joint():
    print 'loading ...'
    with open(cap.absolute_path+'filtered_labeled_'+aimed_file_name, 'rb') as f:
        data = p.load(f)
    print 'done.'
    return data


def load_aimed_pickled_filtered_labeled_sdg_data_joint():
    print 'loading ...'
    with open(cap.absolute_path+'filtered_labeled_sdg_'+aimed_file_name, 'rb') as f:
        data = p.load(f)
    print 'done.'
    return data


def load_chicago_data_joint():
    print 'loading ...'
    file_name = 'chicago_concept_domain_catalyst_data_positive_labels_only.pickle'
    with open(cap.absolute_path+file_name, 'rb') as f:
        data = p.load(f)
    print 'loaded.'
    return data


def load_chicago_sdg_data_joint():
    print 'loading ...'
    file_name = 'concept_domain_catalyst_joint_train_data_dependencies_chicago.pickle'
    with open(cap.absolute_path+file_name, 'rb') as f:
        data = p.load(f)
    print 'done.'
    return data


def load_sentence_id_proteins_list_map():
    with open(cap.absolute_path+'./AImed/aimed_bioc_sentences_proteins.json', 'r') as f:
        sentence_proteins_map = json.load(f)
    return sentence_proteins_map


def load_sentence_id_relations_list_map():
    with open(cap.absolute_path+'./AImed/aimed_bioc_sentence_relations.json', 'r') as f:
        sentence_relations_map = json.load(f)
    return sentence_relations_map


def get_passage_sentence_id_frm_joint_graph_path(joint_graph_path):
    dot_idx = joint_graph_path.index('.dot')
    joint_graph_path = joint_graph_path[:dot_idx]
    dot_idx = None
    prefix = '../../aimed/aimed_bioc_sentences_'
    assert joint_graph_path.startswith(prefix)
    joint_graph_path = joint_graph_path[len(prefix):]
    joint_graph_path_split = joint_graph_path.split('.')
    assert len(joint_graph_path_split) == 2
    passage_sentence_id = joint_graph_path_split[0]
    return passage_sentence_id


def filter_subgraphs_nd_dump():
    data = load_aimed_pickled_data_joint()
    sentence_proteins_map = load_sentence_id_proteins_list_map()
    #
    paths_map = data[gtd.const_paths_map]
    assert paths_map is not None
    interaction_tuples_map = data[gtd.const_interaction_tuples_map]
    assert interaction_tuples_map is not None
    sentences_map = data[gtd.const_sentences_map]
    assert sentences_map is not None
    labels_map = data[gtd.const_joint_labels_map]
    assert labels_map is not None
    #
    list_of_paths = interaction_tuples_map.keys()
    #
    print 'initial no. of graphs before filtering ...', len(list_of_paths)
    #
    for curr_path in list_of_paths:
        if curr_path not in interaction_tuples_map:
            assert curr_path not in paths_map
            assert curr_path not in sentences_map
            assert curr_path not in labels_map
            continue
        #
        is_remove = False
        #
        curr_interaction_tuple = interaction_tuples_map[curr_path]
        #
        if len(curr_interaction_tuple) == 3 and curr_interaction_tuple[1] is None:
            assert curr_interaction_tuple[0] is not None
            assert curr_interaction_tuple[2] is not None
            is_remove = True
        elif len(curr_interaction_tuple) == 4 and curr_interaction_tuple[1] is not None:
            assert curr_interaction_tuple[0] is not None
            assert curr_interaction_tuple[2] is not None
            assert curr_interaction_tuple[3] is not None
            is_remove = True
        else:
            #
            curr_interaction_str_tuple =\
                edk.get_triplet_str_tuple(curr_interaction_tuple, is_concept_mapped_to_interaction=False)
            curr_interaction_tuple = None
            #
            list_of_proteins = list(curr_interaction_str_tuple[1:])
            if None in list_of_proteins:
                list_of_proteins.remove(None)
            #
            curr_sentence_id = get_passage_sentence_id_frm_joint_graph_path(curr_path)
            print 'curr_sentence_id', curr_sentence_id
            assert curr_sentence_id in sentence_proteins_map
            list_gold_proteins = sentence_proteins_map[curr_sentence_id]['proteins_list']
            for curr_protein in list_of_proteins:
                curr_protein = unicode(curr_protein, 'utf-8')
                if not match_protein_name_with_gold_list(curr_protein, list_gold_proteins):
                    # print '{}:{}'.format(curr_protein, list_gold_proteins)
                    print '*************************'
                    print 'curr_protein', curr_protein
                    print 'list_gold_proteins', list_gold_proteins
                    is_remove = True
                    break
        #
        if is_remove:
            paths_map.pop(curr_path, None)
            sentences_map.pop(curr_path, None)
            interaction_tuples_map.pop(curr_path, None)
            if curr_path in labels_map:
                labels_map.pop(curr_path, None)
    #
    print 'no. of graphs after filtering are ', len(paths_map)
    dump_aimed_pickle_data_joint(paths_map, interaction_tuples_map, sentences_map, labels_map, is_filtered=True)


def label_subgraphs_nd_dump():
    data = load_aimed_pickled_filtered_data_joint()
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
    num_positive_labels = 0
    num_negative_labels = 0
    for curr_path in interaction_tuples_map:
        curr_interaction_tuple = interaction_tuples_map[curr_path]
        #
        curr_interaction_str_tuple =\
            edk.get_triplet_str_tuple(curr_interaction_tuple, is_concept_mapped_to_interaction=False)
        curr_interaction_tuple = None
        #
        print 'curr_interaction_str_tuple', curr_interaction_str_tuple
        #
        list_of_proteins = list(curr_interaction_str_tuple[1:])
        if None in list_of_proteins:
            list_of_proteins.remove(None)
        assert len(list_of_proteins) == 2
        #
        curr_sentence_id = get_passage_sentence_id_frm_joint_graph_path(curr_path)
        print 'curr_sentence_id', curr_sentence_id
        assert curr_sentence_id in sentence_relations_map
        list_gold_relation_maps = sentence_relations_map[curr_sentence_id]['relations']
        is_positive_label = False
        for curr_relation_map in list_gold_relation_maps:
            curr_relation_proteins = [curr_relation_map['Arg1'], curr_relation_map['Arg2']]
            if match_protein_name(list_of_proteins[0], curr_relation_proteins[0]) and match_protein_name(list_of_proteins[1], curr_relation_proteins[1]):
                is_positive_label = True
                break
            elif match_protein_name(list_of_proteins[0], curr_relation_proteins[1]) and match_protein_name(list_of_proteins[1], curr_relation_proteins[0]):
                is_positive_label = True
                break
        #
        if is_positive_label:
            labels_map[curr_path] = 1
            num_positive_labels += 1
        else:
            labels_map[curr_path] = 0
            num_negative_labels += 1
    #
    print 'number of positive labeled graphs: ', num_positive_labels
    print 'number of negative labeled graphs: ', num_negative_labels
    print 'total no. of labels: ', num_positive_labels+num_negative_labels
    #
    dump_aimed_pickle_data_joint(paths_map, interaction_tuples_map, sentences_map, labels_map, is_filtered=True, is_labeled=True)


if __name__ == '__main__':
    # generate_subgraphs_nd_dump()
    # filter_subgraphs_nd_dump()
    label_subgraphs_nd_dump()

