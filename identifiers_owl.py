import constants_absolute_path as cap
import csv
import config_hpcc as ch
import json


class ProteinIdsOwl:
    def __init__(self):
        self.identifiers_list = []
        with open(cap.absolute_path+'./csv_biopax_model/ras_2_gold_entities.csv', 'rU') as f:
            reader = csv.DictReader(f)
            for row in reader:
                curr_identifier = row['id'].strip().replace("'", "")
                if curr_identifier:
                    self.identifiers_list.append(curr_identifier)
        #
        if not ch.is_hpcc:
            with open(cap.absolute_path+'../ras_2_gold_entities.json', 'w') as f:
                json.dump(self.identifiers_list, f, ensure_ascii=True, sort_keys=True, indent=5)
        #
