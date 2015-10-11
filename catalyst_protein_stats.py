import constants_absolute_path as cap
import csv
from config_reset_console_output import *


class CatalystProteinStats:
    def __init__(self):
        catalyst_map = {}
        protein_map = {}
        with open(cap.absolute_path+'./json_darpa_model_amrs/interactions.csv', 'rU') as f:
            reader = csv.DictReader(f)
            for row in reader:
                curr_catalyst = row['catalyst']
                if curr_catalyst is not None and curr_catalyst.strip():
                    if curr_catalyst not in catalyst_map:
                        catalyst_map[curr_catalyst] = 1
                    else:
                        catalyst_map[curr_catalyst] += 1
                #
                curr_protein1 = row['protein1']
                if curr_protein1 is not None and curr_protein1.strip():
                    if curr_protein1 not in protein_map:
                        protein_map[curr_protein1] = 1
                    else:
                        protein_map[curr_protein1] += 1
                else:
                    raise AssertionError
                #
                curr_protein2 = row['protein2']
                if curr_protein2 is not None and curr_protein2.strip():
                    if curr_protein2 not in protein_map:
                        protein_map[curr_protein2] = 1
                    else:
                        protein_map[curr_protein2] += 1
        catalyst_prob_map = {}
        for curr_catalyst in catalyst_map:
            num_instances = catalyst_map[curr_catalyst]
            if curr_catalyst in protein_map:
                num_instances += protein_map[curr_catalyst]
            if num_instances <= 5:
                continue
            catalyst_prob_map[curr_catalyst] = catalyst_map[curr_catalyst]/float(num_instances)
        self.catalyst_prob_map = catalyst_prob_map
        print 'self.catalyst_prob_map', self.catalyst_prob_map


if __name__ == '__main__':
    cps_obj = CatalystProteinStats()
