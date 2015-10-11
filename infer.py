raise DeprecationWarning

import genPathwaysModel as gpm
import numpy as np
from config_console_output import *


reload(gpm)


def run(amr_dot_file = './models/bio-mskcc/bio-mskcc_bio.mskcc_0001', num_amrs=None):
    if num_amrs is None or num_amrs == 1:
        amr_dot_files = np.empty(1, dtype=np.object)
        amr_dot_files[0] = amr_dot_file
        # ead.main(amr_dot_file)
    elif num_amrs < 1:
        raise AssertionError
    else:
        amr_dot_files = []
        for i in range(num_amrs):
            try:
                curr_amr_dot_file = amr_dot_file + '.' + str(i+1) + '.dot'
                amr_dot_files.append(curr_amr_dot_file)
            except Exception:
                print 'could not process amr ' + str(i+1)
    amr_dot_file = np.array(amr_dot_files)
    gpm.main_fr_amr(amr_dot_files, amr_dot_file)


if __name__ == "__main__":
    import sys
    run(amr_dot_file=str(sys.argv[1]), num_amrs=int(sys.argv[2]))


