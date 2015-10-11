import train_extractor as te
import pickle
import text_frm_amrs_data as tad


is_load_frm_temp_save = True

temp_amrs_file_path = './temp_amr_graphs_chicago'
amr_text_file_path = '../../amr_text_fr_parsing_sdg/amr_train_data_sentences_chicago'
labeled_data_pickled_file_path = 'labeled_chicago_concept_domain_catalyst_data.pickle'


def get_pickled_data():
    with open(labeled_data_pickled_file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_amr_labels_data():
    if not is_load_frm_temp_save:
        data = get_pickled_data()
        amr_graphs, labels = te.process_pickled_data(data, is_word_vectors=False)
        curr_temp_map = {'amr': amr_graphs, 'label': labels}
        with open(temp_amrs_file_path, 'wb') as f:
            pickle.dump(curr_temp_map, f)
    else:
        with open(temp_amrs_file_path, 'rb') as f:
            curr_temp_map = pickle.load(f)
        amr_graphs = curr_temp_map['amr']
        labels = curr_temp_map['label']
    return amr_graphs, labels


def gen_text_file_frm_train_amrs_data():
    amr_graphs, labels = get_amr_labels_data()
    tad.gen_text_file_frm_train_amrs_data(amr_graphs=amr_graphs,
                                          labels=labels,
                                          amr_text_file_path=amr_text_file_path)


if __name__ == '__main__':
    gen_text_file_frm_train_amrs_data()

