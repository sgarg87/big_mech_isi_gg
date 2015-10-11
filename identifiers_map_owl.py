import constants_absolute_path as cap
import csv
import config_hpcc as ch
import json


class ProteinDisplayGoldIdsOwl:
    def __init__(self):
        self.identifiers_map = {}
        with open(cap.absolute_path+'./csv_biopax_model/ras_2_display_gold.csv', 'rU') as f:
            reader = csv.DictReader(f)
            for row in reader:
                curr_name = row['name'].strip()
                curr_identifier = row['id'].strip()
                if curr_identifier and curr_name:
                    self.identifiers_map[curr_name] = 'Uniprot:'+curr_identifier
        #
        if not ch.is_hpcc:
            with open(cap.absolute_path+'../ras_2_display_gold.json', 'w') as f:
                json.dump(self.identifiers_map, f, ensure_ascii=True, sort_keys=True, indent=5)

    def get_identifier(self, protein_str):
        if protein_str is None or not protein_str.strip():
            return None
        if protein_str in self.identifiers_map:
            return self.identifiers_map[protein_str]
        elif protein_str.lower() in self.identifiers_map:
            return self.identifiers_map[protein_str.lower()]
        elif protein_str.upper() in self.identifiers_map:
            return self.identifiers_map[protein_str.upper()]

