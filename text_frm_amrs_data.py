import train_extractor as te
import numpy as np
import csv
import postprocess_amr_text as pat
import json
import file_paths_extraction as fpe
import preprocess_sentence_fr_dependency_parse as psdp
import codecs
import pickle


is_load_frm_temp_save = True

is_alternative_data = True


def get_temp_amrs_file_path():
    temp_amrs_file_path = './temp_amr_graphs'
    if is_alternative_data:
        temp_amrs_file_path += '_ad'
    return temp_amrs_file_path


def get_amr_text_file_path():
    amr_text_file_path = '../../amr_text_fr_parsing_sdg/amr_train_data_sentences'
    if is_alternative_data:
        amr_text_file_path += '_ad'
    return amr_text_file_path


def get_triplet_str_tuple(curr_triplet_nodes_tuple):
    curr_triplet_str_tuple = []
    m = len(curr_triplet_nodes_tuple)
    for curr_idx in range(m):
        if curr_triplet_nodes_tuple[curr_idx] is not None:
            curr_str = curr_triplet_nodes_tuple[curr_idx].get_name_formatted()
            curr_triplet_str_tuple.append(curr_str)
        else:
            curr_triplet_str_tuple.append(None)
    return tuple(curr_triplet_str_tuple)


def get_amr_labels_data():
    temp_amrs_file_path = get_temp_amrs_file_path()
    if not is_load_frm_temp_save:
        if is_alternative_data:
            amr_graphs, labels = te.get_data_joint(is_train=None, is_word_vectors=False, is_dependencies=False, is_alternative_data=True)
        else:
            amr_graphs, labels = te.get_data_joint(is_train=True, is_word_vectors=False, is_dependencies=False)
        curr_temp_map = {'amr': amr_graphs, 'label': labels}
        with open(temp_amrs_file_path, 'wb') as f:
            pickle.dump(curr_temp_map, f)
    else:
        with open(temp_amrs_file_path, 'rb') as f:
            curr_temp_map = pickle.load(f)
        amr_graphs = curr_temp_map['amr']
        labels = curr_temp_map['label']
    return amr_graphs, labels


def gen_text_file_frm_train_amrs_data(amr_graphs=None, labels=None, amr_text_file_path=None):
    if amr_graphs is None and labels is None:
        amr_graphs, labels = get_amr_labels_data()
    #
    if amr_text_file_path is None:
        amr_text_file_path = get_amr_text_file_path()
    #
    n = amr_graphs.shape[0]
    sentences_map = {}
    #
    f = open(amr_text_file_path+'.txt', 'w')
    csv_f = open(amr_text_file_path+'.csv', 'w')
    field_names = ['text', 'path', 'tuple']
    csv_writer = csv.DictWriter(csv_f, fieldnames=field_names)
    csv_writer.writeheader()
    assertion_count = 0
    for curr_idx in range(n):
        curr_amr_graph_map = amr_graphs[curr_idx, 0]
        curr_label = labels[curr_idx]
        #
        curr_sentence = curr_amr_graph_map['text']
        if curr_sentence is not None:
            print '***********************************'
            curr_sentence = pat.post_process_amr_text_sentence(curr_sentence)
            curr_sentence = psdp.postprocess_sentence_frm_dependency_graph_parse(curr_sentence)
            #
            print curr_sentence
            print type(curr_sentence)
            #
            # if type(curr_sentence) == unicode:
            #     curr_sentence = curr_sentence.encode('utf8')
            #     print 'utf changes ...'
            #     print curr_sentence
            #     print type(curr_sentence)
            # elif type(curr_sentence) == str:
            #     pass
            # else:
            #     raise AssertionError
            #
            curr_path = curr_amr_graph_map['path']
            #
            print curr_path
            #
            curr_interaction_triplet_str_tuple = get_triplet_str_tuple(curr_amr_graph_map['tuple'])
            #
            print curr_interaction_triplet_str_tuple
            #
            csv_writer.writerow({'text': curr_sentence, 'path': curr_path, 'tuple': curr_interaction_triplet_str_tuple})
            #
            if curr_sentence not in sentences_map:
                f.write(curr_sentence+'\n')
                sentences_map[curr_sentence] = {}
            sentences_map[curr_sentence][curr_path] = {'str_tuple': curr_interaction_triplet_str_tuple, 'label': curr_label}
            #
            org_amr_dot_file_path = fpe.extract_original_amr_dot_file_name(curr_path)
            print 'org_amr_dot_file_path', org_amr_dot_file_path
            if 'org_amr_dot_file_path' not in sentences_map[curr_sentence]:
                sentences_map[curr_sentence]['org_amr_dot_file_path'] = org_amr_dot_file_path
            else:
                if sentences_map[curr_sentence]['org_amr_dot_file_path'] != org_amr_dot_file_path:
                    assertion_count += 1
                    # if assertion_count > 500:
                    #     raise AssertionError(sentences_map[curr_sentence]['org_amr_dot_file_path'])
    f.close()
    csv_f.close()
    #
    with open(amr_text_file_path+'.json', 'w') as json_f:
        json.dump(sentences_map, json_f, sort_keys=True, indent=5, ensure_ascii=False)


if __name__ == '__main__':
    gen_text_file_frm_train_amrs_data()

