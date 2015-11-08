import numpy as np
import random as r
r.seed(871227)
#
import validate_on_chicago_data_full_scale as vcdfs
import json


if __name__ == '__main__':
    is_filter_chicago_interactions_on_proteins_nd_interaction_types_in_extracted = True
    is_filter_on_positive_interactions_in_chicago = True
    is_positive = False
    is_filter = False
    #
    if is_filter:
        extracted_interactions_list = np.load('./amr_graphs_chicago_filtered_str_tuples.npz')['arr_0']
    else:
        extracted_interactions_list = np.load('./chicago_interactions_str_list.npz')['arr_0']
    print 'extracted_interactions_list.shape', extracted_interactions_list.shape
    #
    #original in chicago
    chicago_sentence_id__interactions_list_map \
        = vcdfs.get_positive_sentence__interactions_list_map(is_filter=is_filter, is_positive=is_positive)
    #
    if is_filter_chicago_interactions_on_proteins_nd_interaction_types_in_extracted:
        sentence_id_proteins_list_map_in_extracted_interactions \
            = vcdfs.get_list_of_proteins_in_extracted_interactions_str_tuples(extracted_interactions_list.tolist())
        #
        sentence_id_interaction_types_map_in_extracted_interactions \
            = vcdfs.get_sentence_id_interaction_types_list_map_in_extracted_interactions_str_tuples(extracted_interactions_list.tolist())
        #
        chicago_sentence_id__interactions_list_map \
            = vcdfs.filter_sentence_id_interactions_list_map_wd_proteins_list(
                chicago_sentence_id__interactions_list_map,
                sentence_id_proteins_list_map_in_extracted_interactions,
                sentence_id_interaction_types_map_in_extracted_interactions)
        #
        print chicago_sentence_id__interactions_list_map.keys()
        print 'len(chicago_sentence_id__interactions_list_map.keys())', len(chicago_sentence_id__interactions_list_map.keys())
        #
        with open('./chicago_sentence_ids_with_interactions_matching_isi_format.json', 'w') as f:
            json.dump(chicago_sentence_id__interactions_list_map.keys(), f, indent=4)

