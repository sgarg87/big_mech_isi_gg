import semi_automated_extraction_features_chicago_data as saefcd
import validate_on_chicago_data_full_scale as vcdfs
import eval_divergence_frm_kernel as edk
import constants_absolute_path as cap
import gen_extractor_features_data as gtd
import pickle as p
import copy


def remove_key_frm_maps(paths_map, interaction_tuples_map, sentences_map, key):
    paths_map.pop(key, None)
    interaction_tuples_map.pop(key, None)
    sentences_map.pop(key, None)


def filter_data(data):
    paths_map = data[gtd.const_paths_map]
    print len(paths_map)
    interaction_tuples_map = data[gtd.const_interaction_tuples_map]
    print len(interaction_tuples_map)
    sentences_map = data[gtd.const_sentences_map]
    print len(sentences_map)
    labels_map = data[gtd.const_joint_labels_map]
    print len(labels_map)
    #
    chicago_sentence_id__interactions_list_map = vcdfs.get_positive_sentence__interactions_list_map(is_filter=False)
    print len(chicago_sentence_id__interactions_list_map), chicago_sentence_id__interactions_list_map
    #
    paths_list = copy.copy(paths_map.keys())
    #
    for curr_path in paths_list:
        print 'curr_path', curr_path
        assert curr_path in interaction_tuples_map
        assert curr_path in sentences_map
        #
        curr_sentence_id = saefcd.get_passage_sentence_id_frm_joint_graph_path(curr_path)
        curr_sentence_id = 'ID'+str(curr_sentence_id)
        #
        if curr_sentence_id not in chicago_sentence_id__interactions_list_map:
            remove_key_frm_maps(paths_map, interaction_tuples_map, sentences_map, curr_path)
            continue
        #
        curr_interaction_tuple = interaction_tuples_map[curr_path]
        curr_interaction_tuple_str \
            = edk.get_triplet_str_tuple(curr_interaction_tuple, is_concept_mapped_to_interaction=False)
        curr_interaction_tuple = None
        print 'curr_interaction_tuple_str', curr_interaction_tuple_str
        #
        if len(curr_interaction_tuple_str) == 3:
            if curr_interaction_tuple_str[1] is None: #catalyst is None
                remove_key_frm_maps(paths_map, interaction_tuples_map, sentences_map, curr_path)
                continue
        elif len(curr_interaction_tuple_str) == 4:
            if curr_interaction_tuple_str[1] is not None:
                remove_key_frm_maps(paths_map, interaction_tuples_map, sentences_map, curr_path)
                continue
        else:
            raise AssertionError
        #
        if len(curr_interaction_tuple_str) == 3:
            curr_proteins = list(curr_interaction_tuple_str[1:])
        elif len(curr_interaction_tuple_str) == 4:
            curr_proteins = list(curr_interaction_tuple_str[2:])
        else:
            raise AssertionError
        #
        assert None not in curr_proteins
        print 'chicago_sentence_id__interactions_list_map[curr_sentence_id]', chicago_sentence_id__interactions_list_map[curr_sentence_id]
        is_match = False
        for curr_chicago_interaction in chicago_sentence_id__interactions_list_map[curr_sentence_id]:
            print 'curr_chicago_interaction', curr_chicago_interaction
            #
            curr_chicago_interaction_proteins = curr_chicago_interaction[1:]
            #
            if not saefcd.match_protein_name(curr_interaction_tuple_str[0], curr_chicago_interaction[0]):
                remove_key_frm_maps(paths_map, interaction_tuples_map, sentences_map, curr_path)
                continue
            #
            if not (saefcd.match_protein_name(curr_proteins[0], curr_chicago_interaction_proteins[0])
                    and saefcd.match_protein_name(curr_proteins[1], curr_chicago_interaction_proteins[1])):
                if len(curr_interaction_tuple_str) == 3:
                    remove_key_frm_maps(paths_map, interaction_tuples_map, sentences_map, curr_path)
                    continue
                elif len(curr_interaction_tuple_str) == 4:
                    if not (saefcd.match_protein_name(curr_proteins[0], curr_chicago_interaction_proteins[1])
                        and saefcd.match_protein_name(curr_proteins[1], curr_chicago_interaction_proteins[0])):
                        remove_key_frm_maps(paths_map, interaction_tuples_map, sentences_map, curr_path)
                        continue
                else:
                    raise AssertionError
            #
            is_match = True
            break
        #
        if not is_match:
            remove_key_frm_maps(paths_map, interaction_tuples_map, sentences_map, curr_path)
            continue
    #
    data = {}
    data[gtd.const_paths_map] = paths_map
    data[gtd.const_interaction_tuples_map] = interaction_tuples_map
    data[gtd.const_sentences_map] = sentences_map
    data[gtd.const_joint_labels_map] = labels_map
    return data


if __name__ == '__main__':
    paths_map = {}
    interaction_tuples_map = {}
    sentences_map = {}
    labels_map = {}
    num_cores = 30
    print 'num_cores', num_cores
    for curr_core in range(num_cores):
        if curr_core == 2:
            continue
        #
        print 'curr_core', curr_core
        #
        file_name = saefcd.get_file_path(num_cores, curr_core)
        print 'file_name', file_name
        with open(cap.absolute_path+file_name, 'rb') as f:
            print 'loading ...'
            curr_data = p.load(f)
            print 'loaded'
            #
            print 'filtering ...'
            curr_data = filter_data(curr_data)
            print 'filtered'
            #
            print 'adding to the global map ...'
            paths_map.update(curr_data[gtd.const_paths_map])
            print len(paths_map)
            interaction_tuples_map.update(curr_data[gtd.const_interaction_tuples_map])
            print len(interaction_tuples_map)
            sentences_map.update(curr_data[gtd.const_sentences_map])
            print len(sentences_map)
            labels_map.update(curr_data[gtd.const_joint_labels_map])
            print len(labels_map)
            print 'added'
    #
    assert not labels_map, 'labels map is empty so far'
    for curr_path in paths_map:
        labels_map[curr_path] = 1
    #
    assert len(paths_map) == len(labels_map)
    #
    data = {}
    data[gtd.const_paths_map] = paths_map
    data[gtd.const_interaction_tuples_map] = interaction_tuples_map
    data[gtd.const_sentences_map] = sentences_map
    data[gtd.const_joint_labels_map] = labels_map
    #
    new_path = saefcd.get_file_path(None, None)
    new_path += 'positive_labels_only'
    #
    with open(cap.absolute_path+new_path, 'wb') as f:
        p.dump(data, f)
