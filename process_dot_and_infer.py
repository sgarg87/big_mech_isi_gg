import extract_from_amr_dot as ead
import genPathwaysModel as gpm
import time
import config_extraction as ce
import re
import pickle
import constants_absolute_path as cap
import preprocess_amr_dots as pad
import config_json_save as cjs
import interactions_to_json_or_amrs as ija
import ntpath
from config_console_output import *
import config_processing as cp


is_biopax_model = True
if is_biopax_model:
    import biopax_model_obj as bmo

if ce.is_kernel:
    import gen_interactions_wd_kernels as gik

reload(ead)
reload(gpm)

# in seconds
computation_time_to_parse_per_amr_frm_text = 11
is_preprocess = True


def save_json_objs_as_index_cards(json_objs_list, joint_file_path):
    ija_obj = ija.Interaction_to_JSON()
    ija_obj.write_json_index_cards(json_objs_list, ntpath.split(joint_file_path)[0]+'/index_cards')


def get_json_objs_frm_interactions(interaction_objs_map, model_json_objs, protein_name_idlist_map, name_identifier_map_amrs, amr_parsing_frm_txt_compute_time, extraction_compute_time, pmc_id):
    if 'complex' not in interaction_objs_map or not interaction_objs_map['complex']:
        print 'No complex formation type interactions.'
    ija_obj = ija.Interaction_to_JSON()
    json_objs = ija_obj.interaction_obj_to_darpa_json\
        (interaction_objs_map['state_change']+interaction_objs_map['complex'], amr_parsing_frm_txt_compute_time=amr_parsing_frm_txt_compute_time,
         extraction_compute_time=extraction_compute_time, pmc_id=pmc_id, protein_name_idlist_map=protein_name_idlist_map,
         name_identifier_map_amrs=name_identifier_map_amrs)
    ija_obj.update_relation_model(json_objs, model_json_objs)
    return json_objs


def pickle_interaction_objs(interaction_objs_map, joint_file_path):
    with open(cap.absolute_path+joint_file_path+'.pickle', 'wb') as f:
        pickle.dump(interaction_objs_map, f)


def extract_pmc_id(amr_dot_file, passage_or_pmc):
    print 'amr_dot_file', amr_dot_file
    print 'passage_or_pmc', passage_or_pmc
    pmc_id_list = re.findall(r'pmid_\d+_\d+_\d+', amr_dot_file)
    if not pmc_id_list:
        pmc_id_list = re.findall(r'pmid_\d+_\d+', amr_dot_file)
    if pmc_id_list is not None and pmc_id_list:
        if len(pmc_id_list) > 1:
            raise AssertionError
        return pmc_id_list[0].replace('pmid', 'PMC').replace('_', '')
    pmc_id_list = re.findall(r'PMC.+', passage_or_pmc)
    if pmc_id_list is not None and pmc_id_list:
        if len(pmc_id_list) > 1:
            raise AssertionError
        return pmc_id_list[0]


def get_model():
    return bmo.bm_obj.json_objs, bmo.bm_obj.protein_identifier_list_map


def run(amr_dot_file=None, start_amr=None, end_amr=None, model_file=None, passage_or_pmc=None, entities_info_file=None):
    if not cp.is_processing_amrs:
        raise AssertionError
    # gen candidate interactions from amrs
    start_time = time.time()
    # get model
    model_json_objs, protein_name_idlist_map = get_model()
    #
    if is_preprocess:
        name_identifier_map_amrs = pad.run(amr_dot_file, start_amr, end_amr)
    else:
        name_identifier_map_amrs = None
    #
    pmc_id = extract_pmc_id(amr_dot_file, passage_or_pmc)
    print 'extracted pmc_id is ', pmc_id
    #
    amr_parsing_frm_txt_compute_time = (end_amr-start_amr)*computation_time_to_parse_per_amr_frm_text
    print 'amr_parsing_frm_txt_compute_time', amr_parsing_frm_txt_compute_time
    #
    joint_file_path = amr_dot_file + '_' + str(start_amr) + '_' + str(end_amr)
    #
    amr_dot_files = []
    interaction_objs_map = {}
    interaction_objs_map['state_change'] = []
    interaction_objs_map['complex'] = []
    interaction_objs_map['compute_time'] = None
    json_objs_list = []
    count = 0
    for i in range(start_amr, end_amr+1):
        count += 1
        curr_amr_dot_file = amr_dot_file + '.' + str(i) + '.dot'
        print 'curr_amr_dot_file:', curr_amr_dot_file
        #
        curr_interaction_objs_map = gik.gen_data_features_fr_pathway_modeling(curr_amr_dot_file, protein_name_idlist_map=protein_name_idlist_map)
        print 'curr_interaction_objs_map', curr_interaction_objs_map
        extraction_compute_time = time.time()-start_time
        #
        interaction_objs_map['state_change'] += curr_interaction_objs_map['state_change']
        interaction_objs_map['complex'] += curr_interaction_objs_map['complex']
        interaction_objs_map['compute_time'] = extraction_compute_time
        #
        curr_json_objs = get_json_objs_frm_interactions(curr_interaction_objs_map, model_json_objs, protein_name_idlist_map, name_identifier_map_amrs,
                                                        amr_parsing_frm_txt_compute_time, extraction_compute_time, pmc_id)
        print 'curr_json_objs', curr_json_objs
        json_objs_list += curr_json_objs
        #
        if (count % cjs.num_interactions_every_save) == 0:
            pickle_interaction_objs(interaction_objs_map, joint_file_path)
            save_json_objs_as_index_cards(json_objs_list, joint_file_path)
        #
        amr_dot_files.append(curr_amr_dot_file)
    #
    pickle_interaction_objs(interaction_objs_map, joint_file_path)
    save_json_objs_as_index_cards(json_objs_list, joint_file_path)
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
        run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]), model_file=str(sys.argv[4]), passage_or_pmc=str(sys.argv[5]))
    elif len(sys.argv) == 7:
        run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]), model_file=str(sys.argv[4]), passage_or_pmc=str(sys.argv[5]), entities_info_file=str(sys.argv[6]))
    else:
        raise AssertionError
