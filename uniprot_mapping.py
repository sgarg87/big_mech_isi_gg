import csv
import re
import time
import constants_absolute_path as cap
import json
from config_console_output import *
import config_hpcc as ch


class UniprotMapping:
    def __init__(self):
        start_time = time.time()
        self.numeric_to_str_map = {}
        with open(cap.absolute_path+'./csv_biopax_model/uniprot_mappings/uniprot_sprot_ID_AC.csv', 'rU') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.numeric_to_str_map[row['id'].strip()] = row['str'].strip()
        #
        if not ch.is_hpcc:
            with open(cap.absolute_path+'../uniprot_mappings.json', 'w') as f:
                json.dump(self.numeric_to_str_map, f, ensure_ascii=True, sort_keys=True, indent=5)
        #
        print 'Time to load the uniprot mapping was ', time.time() - start_time

    def get_mapping(self, uniprot_id):
        if uniprot_id is None or not uniprot_id.lower():
            return None
        if 'uniprot' not in uniprot_id.lower():
            return None
        numeric = re.findall(r'Uniprot:(.+?) ', uniprot_id+' ')
        if len(numeric) == 1:
            id = numeric[0]
            if id in self.numeric_to_str_map:
                return 'Uniprot:'+self.numeric_to_str_map[id]

