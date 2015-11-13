import extract_from_amr_dot as ead
import genPathwaysModel as gpm
import time
import config_extraction as ce
import pickle
import constants_absolute_path as cap
import preprocess_amr_dots as pad
import config_json_save as cjs
import process_dot_and_infer as pdi
import config_processing as cp
# from config_reset_console_output import *
import post_process_json_darpa as ppjd

is_biopax_model = True
if is_biopax_model:
    import biopax_model_obj as bmo

if ce.is_kernel:
    import gen_interactions_wd_kernels as gik

reload(ead)
reload(gpm)

# in seconds
computation_time_to_parse_per_amr_frm_text = 11

# in seconds
compute_time_to_extract_per_amr = 30
#
is_preprocess = False


def load_pickled_interaction_objs(joint_file_path):
    with open(cap.absolute_path+joint_file_path+'.pickle', 'r') as f:
        interaction_objs_map = pickle.load(f)
    return interaction_objs_map


def run(amr_dot_file=None, start_amr=None, end_amr=None, model_file=None, passage_or_pmc=None, entities_info_file=None):
    if cp.is_processing_amrs:
        raise AssertionError
    # gen candidate interactions from amrs
    start_time = time.time()
    # get model
    model_json_objs, protein_name_idlist_map = pdi.get_model()
    print 'Time to get the model was ', time.time()-start_time
    #
    start_time = time.time()
    if is_preprocess:
        name_identifier_map_amrs = pad.run(amr_dot_file, start_amr, end_amr)
    else:
        name_identifier_map_amrs = None
    print 'Time to preprocess the AMRs was ', time.time()-start_time
    #
    # pmc_id = pdi.extract_pmc_id(amr_dot_file, passage_or_pmc)
    pmc_id = pdi.extract_PMC(amr_dot_file)
    print 'extracted pmc_id is ', pmc_id
    #
    amr_parsing_frm_txt_compute_time = (end_amr-start_amr)*computation_time_to_parse_per_amr_frm_text
    print 'amr_parsing_frm_txt_compute_time', amr_parsing_frm_txt_compute_time
    #
    joint_file_path = amr_dot_file + '_' + str(start_amr) + '_' + str(end_amr)
    #
    start_time = time.time()
    interaction_objs_map = load_pickled_interaction_objs(joint_file_path)
    print 'interaction_objs_map', interaction_objs_map
    #
    if 'compute_time' in interaction_objs_map:
        extraction_compute_time = interaction_objs_map['compute_time']
    else:
        extraction_compute_time = (end_amr-start_amr)*compute_time_to_extract_per_amr
    print 'extraction_compute_time', extraction_compute_time
    json_objs_list = pdi.get_json_objs_frm_interactions(
        interaction_objs_map, model_json_objs, protein_name_idlist_map, name_identifier_map_amrs,
        amr_parsing_frm_txt_compute_time, extraction_compute_time, pmc_id)
    #
    # post process json objects
    ppjd_obj = ppjd.PostProcessJSONDARPA(json_objs_list=json_objs_list)
    ppjd_obj.remove_identifier_prob_list()
    ppjd_obj.filter_out_symmetric_duplicates_fr_evidence()
    ppjd_obj.filter_out_increase_decrease_wd_no_catalyst()
    ppjd_obj.filter_out_duplicates_fr_evidence(is_set=True)
    # ppjd_obj.swap_catalyst_protein()
    # json_objs_list = ppjd_obj.filter_top_fr_interaction_type_in_amr()
    #
    pdi.save_json_objs_as_index_cards(json_objs_list, joint_file_path)
    #
    print 'Total computation time was ', time.time()-start_time


if __name__ == "__main__":
    import sys
    print sys.argv
    #zeroth argument is file python code file path
    if len(sys.argv) == 2:
        run(amr_dot_file=str(sys.argv[1]))
    elif len(sys.argv) == 4:
        run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]))
    elif len(sys.argv) == 5:
        run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]), model_file=str(sys.argv[4]))
    elif len(sys.argv) == 6:
        run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]), model_file=str(sys.argv[4]),
            passage_or_pmc=str(sys.argv[5]))
    elif len(sys.argv) == 7:
        run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]), model_file=str(sys.argv[4]),
            passage_or_pmc=str(sys.argv[5]), entities_info_file=str(sys.argv[6]))
    else:
        raise AssertionError

