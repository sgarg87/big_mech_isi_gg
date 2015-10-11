import glob
import pydot as pd
import constants_absolute_path as cap
import postprocess_amr_text as pat
import preprocess_sentence_fr_dependency_parse as psdp
import file_paths_extraction as fpe
import json


def gen_txt_frm_dot(folder_path, text_file_path):
    dot_files_paths_list = glob.glob(folder_path+"*.dot")
    print 'len(dot_files_paths_list)', len(dot_files_paths_list)
    #
    f_text = open(text_file_path+'.txt', 'w')
    sentences_map = {}
    for curr_dot_file_path in dot_files_paths_list:
        curr_amr_obj = pd.graph_from_dot_file(cap.absolute_path+curr_dot_file_path)
        curr_sentence = curr_amr_obj.get_label()
        #
        curr_sentence = pat.post_process_amr_text_sentence(curr_sentence)
        curr_sentence = psdp.postprocess_sentence_frm_dependency_graph_parse(curr_sentence)
        f_text.write(curr_sentence+'\n')
        #
        if curr_sentence not in sentences_map:
            sentences_map[curr_sentence] = curr_dot_file_path
        else:
            continue
    f_text.close()
    with open(text_file_path+'.json', 'w') as json_f:
        json.dump(sentences_map, json_f, sort_keys=True, indent=5, ensure_ascii=False)


if __name__ == '__main__':
    import sys
    folder_path = sys.argv[1]
    text_file_path = sys.argv[2]
    gen_txt_frm_dot(folder_path, text_file_path)

