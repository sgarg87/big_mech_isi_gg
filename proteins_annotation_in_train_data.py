import train_extractor as te
import gen_extractor_features_data as gtd
import eval_divergence_frm_kernel as edk
import json
import constants_absolute_path as cap
import pickle
import difflib as dl


file_name = './protein_sentences_map_train_data.json'


def get_unique_protein_sentences_map_nd_dump_json():
    is_load_amr_data = True
    amr_pickle_file_path = './amr_data_temp.pickle'
    with open(cap.absolute_path+amr_pickle_file_path, 'rb') as f_p:
        print 'loading ...'
        amr_data = pickle.load(f_p)
        print 'done'
        amr_graphs = amr_data['amr']
        labels = amr_data['label']
    #
    protein_sentences_map = {}
    #
    n = len(amr_graphs)
    print 'n', n
    for i in range(n):
        curr_amr_graph = amr_graphs[i, 0]
        curr_path = curr_amr_graph['path']
        curr_interaction_tuple = curr_amr_graph['tuple']
        curr_sentence = curr_amr_graph['text']
        curr_label = labels[i]
        curr_interaction_str_tuple = edk.get_triplet_str_tuple(curr_interaction_tuple, is_concept_mapped_to_interaction=False)
        curr_interaction_tuple = None
        #
        list_of_proteins = list(curr_interaction_str_tuple[1:])
        assert len(list_of_proteins) in [2, 3]
        #
        if None in list_of_proteins:
            list_of_proteins.remove(None)
        for curr_protein in list_of_proteins:
            # curr_protein = unicode(curr_protein, 'utf-8')
            if curr_protein not in protein_sentences_map:
                protein_sentences_map[curr_protein] = {}
                protein_sentences_map[curr_protein]['text'] = []
                if curr_label != 0:
                    protein_sentences_map[curr_protein]['label'] = 1
            #
            if curr_sentence not in protein_sentences_map[curr_protein]['text']:
                protein_sentences_map[curr_protein]['text'].append(curr_sentence)
    #
    assert protein_sentences_map is not None and protein_sentences_map
    #
    print 'no of proteins', len(protein_sentences_map)
    #
    dump_protein_sentences_map(protein_sentences_map)


def dump_protein_sentences_map(protein_sentences_map):
    with open(cap.absolute_path+file_name, 'w') as f:
        json.dump(protein_sentences_map, f, indent=4)


def load_protein_sentences_map():
    with open(cap.absolute_path+file_name, 'r') as f:
        protein_sentences_map = json.load(f)
    assert protein_sentences_map is not None and protein_sentences_map
    return protein_sentences_map


def label_proteins():
    def input_protein_valid_label():
        try:
            curr_label = raw_input('Enter invalid-0, valid-1:')
            curr_label = int(curr_label)
            if curr_label < 0 or curr_label > 1:
                raise ValueError
            print '***labeling input successful: ', curr_label
        except ValueError:
            return input_protein_valid_label()
        return curr_label

    protein_sentences_map = load_protein_sentences_map()
    for curr_protein in protein_sentences_map:
        curr_protein_map = protein_sentences_map[curr_protein]
        if 'label' in curr_protein_map:
            assert curr_protein_map['label'] == 1
        else:
            print '****************************'
            for curr_sentence in protein_sentences_map[curr_protein]['text']:
                print curr_sentence
            print '****************************'
            print 'curr_protein', curr_protein
            curr_label = input_protein_valid_label()
            curr_protein_map['label'] = curr_label
            #
            dump_protein_sentences_map(protein_sentences_map)
    #
    valid_proteins_list = []
    for curr_protein in protein_sentences_map:
        curr_protein_map = protein_sentences_map[curr_protein]
        assert 'label' in curr_protein_map
        curr_label = curr_protein_map['label']
        if curr_label == 1:
            valid_proteins_list.append(curr_protein)
    #
    with open(cap.absolute_path+'./valid_proteins_in_train_data.json', 'w') as f:
        json.dump(valid_proteins_list, f, indent=4)
    #
    # print valid_proteins_list
    print 'No. of valid proteins', len(valid_proteins_list)


if __name__ == '__main__':
    pass
