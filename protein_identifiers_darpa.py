import csv
import re
import time
import constants_absolute_path as cap
import uniprot_mapping_obj as umo
import json
from config_console_output import *
import config_hpcc as ch


class ProteinIdentifiersDARPA:
    def __init__(self):
        start_time = time.time()
        self.name_identifier_list_map = {}
        with open(cap.absolute_path+'./csv_biopax_model/protein_identifiers_darpa.csv', 'rU') as f:
            reader = csv.DictReader(f)
            for row in reader:
                identifier_list_str = row['UNIFICATION_XREF']
                if identifier_list_str is not None and identifier_list_str.strip():
                    identifier_list = identifier_list_str.split(';')
                    for curr_identifier in identifier_list:
                        if 'null' in curr_identifier.lower():
                            identifier_list.remove(curr_identifier)
                    if identifier_list:
                        for curr_idx in range(len(identifier_list)):
                            curr_identifier = identifier_list[curr_idx]
                            curr_identifier_uniprot_mapping = umo.um_obj.get_mapping(curr_identifier)
                            if curr_identifier_uniprot_mapping is not None and curr_identifier_uniprot_mapping.strip():
                                identifier_list[curr_idx] = curr_identifier_uniprot_mapping
                        identifier_list = list(set(identifier_list))
                        self.name_identifier_list_map[row['PARTICIPANT'].strip()] = identifier_list
        #
        if not ch.is_hpcc:
            with open(cap.absolute_path+'../name_identifier_list_map_darpa.json', 'w') as f:
                json.dump(self.name_identifier_list_map, f, ensure_ascii=True, sort_keys=True, indent=5)
        #
        print 'Time to load the protein identifiers was ', time.time() - start_time

    def get_identifier(self, protein_str):
        if protein_str is None or not protein_str.strip():
            return None
        if protein_str in self.name_identifier_list_map:
            return ','.join(self.name_identifier_list_map[protein_str])
        elif protein_str.lower() in self.name_identifier_list_map:
            return ','.join(self.name_identifier_list_map[protein_str.lower()])
