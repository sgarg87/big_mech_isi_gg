import csv
import os
import glob
import json
import constants_darpa_json_format as cdjf
import difflib as dl
import postprocess_amr_text as pat


def get_sentence_id_text_map():
    # field_names = ['PMC_ID', 'Sent_ID', 'Sentence', 'Paragraph']
    sentence_text_map = {}
    file = '../../../mitre_reeval/human_curated_evidence_matched_unique.csv'
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for curr_row in reader:
            count += 1
            curr_sentence_id = curr_row['Sent_ID']
            curr_sentence = curr_row['Sentence']
            #
            assert curr_sentence_id not in sentence_text_map
            sentence_text_map[curr_sentence_id] = curr_sentence
    #
    print count
    print len(sentence_text_map)
    print sentence_text_map.keys()
    return sentence_text_map


def get_sentence_ids_list(text, sentence_text_map):
    #
    ids_list = []
    #
    for curr_sentence_id in sentence_text_map:
        dl_obj = dl.SequenceMatcher(None, text, sentence_text_map[curr_sentence_id])
        curr_ratio = dl_obj.quick_ratio()
        #
        if curr_ratio > 0.94:
            ids_list.append(curr_sentence_id)
    #
    assert ids_list
    assert len(ids_list) <= 2
    #
    return ids_list


def update_index_cards(sentence_text_map):
    path = '../../../mitre_reeval/temp/index_cards'
    pmc_dirs = [x[0] for x in os.walk(path)][1:]
    print len(pmc_dirs)
    #
    for curr_pmc_dir in pmc_dirs:
        # print 'curr_pmc_dir', curr_pmc_dir
        index_card_paths = glob.glob(curr_pmc_dir+"/*.json")
        assert index_card_paths
        # print index_card_paths
        #
        for curr_index_card_path in index_card_paths:
            with open(curr_index_card_path, 'r') as idx_f:
                curr_index_card_json_obj = json.load(idx_f)
                assert len(curr_index_card_json_obj[cdjf.evidence]) == 1
                curr_text = curr_index_card_json_obj[cdjf.evidence][0]
                curr_text = pat.post_process_amr_text_sentence(curr_text)
                # print 'curr_text', curr_text
                #
                curr_ids_list = get_sentence_ids_list(curr_text, sentence_text_map)
                assert curr_ids_list
                print 'curr_ids_list', curr_ids_list
                #
                curr_index_card_json_obj['sentence_ids_list'] = curr_ids_list
            #
            with open(curr_index_card_path, 'w') as idx_f:
                json.dump(curr_index_card_json_obj, idx_f, indent=4, sort_keys=True)


if __name__ == '__main__':
    sentence_text_map = get_sentence_id_text_map()
    #
    update_index_cards(sentence_text_map)
