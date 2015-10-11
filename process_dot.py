import extract_from_amr_dot as ead
import genPathwaysModel as gpm
import numpy as np
import time
from config_console_output import *


reload(ead)
reload(gpm)


def run(amr_dot_file='./models/bio-mskcc/bio-mskcc_bio.mskcc_0001', start_amr=None, end_amr=None, model_file=None, passage_name=None, entities_info_file=None):
    start_time = time.time()
    if start_amr is None or end_amr is None:
        amr_dot_files = np.empty(1, dtype=np.object)
        amr_dot_files[0] = amr_dot_file
        ead.main(amr_dot_file)
    elif start_amr >= 0 and end_amr >= 0:
        amr_dot_files = []
        for i in range(start_amr, end_amr+1):
            # try:
            curr_amr_dot_file = amr_dot_file + '.' + str(i) + '.dot'
            print 'curr_amr_dot_file:', curr_amr_dot_file
            ead.main(curr_amr_dot_file)
            amr_dot_files.append(curr_amr_dot_file)
            # except Exception as e:
            #     print 'could not process amr ' + str(i+1)
            #     print e
    else:
        raise AssertionError
    # amr_dot_files_arr = np.array(amr_dot_files)
    if model_file is not None:
        model_interactions = ead.parse_interactions(model_file)
    else:
        model_interactions = None
    if entities_info_file is not None:
        protein_identifier_map, protein_family_map = gpm.parse_entities_info(entities_info_file)
    else:
        protein_identifier_map, protein_family_map = None, None
    if passage_name is not None:
        gpm.main_fr_amr(amr_dot_files, amr_dot_file, True, model_interactions, passage_name=passage_name,
                        result_file_path=amr_dot_file+passage_name, model_output_file_path=amr_dot_file+passage_name+'_model',
                        protein_identifier_map=protein_identifier_map, protein_family_map=protein_family_map, is_infer=False)
    else:
        if start_amr is not None and end_amr is not None:
            gpm.main_fr_amr(amr_dot_files, amr_dot_file, True, model_interactions, result_file_path=amr_dot_file+'_'+str(start_amr)+'_'+str(end_amr), model_output_file_path=amr_dot_file+'_'+str(start_amr)+'_'+str(end_amr)+'_model', protein_identifier_map=protein_identifier_map, protein_family_map=protein_family_map, is_infer=False)
        else:
            gpm.main_fr_amr(amr_dot_files, amr_dot_file, True, model_interactions, protein_identifier_map=protein_identifier_map, protein_family_map=protein_family_map, is_infer=False)
    print 'Time to process and infer is:', time.time()-start_time


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
        run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]), model_file=str(sys.argv[4]), passage_name=str(sys.argv[5]))
    elif len(sys.argv) == 7:
        run(amr_dot_file=str(sys.argv[1]), start_amr=int(sys.argv[2]), end_amr=int(sys.argv[3]), model_file=str(sys.argv[4]), passage_name=str(sys.argv[5]), entities_info_file=str(sys.argv[6]))
    else:
        raise AssertionError

