import time
import process_dot_and_infer as pdi
import constants_absolute_path as cap
import pickle


def load_pickled_interaction_objs(joint_file_path):
    with open(cap.absolute_path+joint_file_path+'.pickle', 'r') as f:
        interaction_objs_map = pickle.load(f)
    return interaction_objs_map


# in seconds
compute_time_to_extract_per_amr = 260


def run(amr_dot_file=None, start_amr=None, end_amr=None, model_file=None, passage_or_pmc=None):
    # gen candidate interactions from amrs
    start_time = time.time()
    # get model
    model_json_objs, protein_name_idlist_map = pdi.get_model()
    name_identifier_map_amrs = None
    #
    pmc_id = pdi.extract_pmc_id(amr_dot_file, passage_or_pmc)
    print 'extracted pmc_id is ', pmc_id
    #
    amr_parsing_frm_txt_compute_time = (end_amr-start_amr)*pdi.computation_time_to_parse_per_amr_frm_text
    print 'amr_parsing_frm_txt_compute_time', amr_parsing_frm_txt_compute_time
    #
    joint_file_path = amr_dot_file + '_' + str(start_amr) + '_' + str(end_amr)
    interaction_objs_map = load_pickled_interaction_objs(joint_file_path)
    if 'compute_time' in interaction_objs_map:
        extraction_compute_time = interaction_objs_map['compute_time']
    else:
        extraction_compute_time = (end_amr-start_amr)*compute_time_to_extract_per_amr
    print 'extraction_compute_time', extraction_compute_time
    #
    json_objs_list = pdi.get_json_objs_frm_interactions(interaction_objs_map, model_json_objs, protein_name_idlist_map, name_identifier_map_amrs,
                                                    amr_parsing_frm_txt_compute_time, extraction_compute_time, pmc_id)
    #
    pdi.save_json_objs_as_index_cards(json_objs_list, joint_file_path)


if __name__ == "__main__":
    import sys
    run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]), model_file=str(sys.argv[4]), passage_or_pmc=str(sys.argv[5]))

