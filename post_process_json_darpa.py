import constants_darpa_json_format as cdjf
import darpa_interaction_types as dit
from config_reset_console_output import *
import catalyst_protein_stats as cps
import json


class PostProcessJSONDARPA:
    def __init__(self, json_objs_list):
        self.json_objs_list = json_objs_list

    def filter_out_duplicates_fr_evidence(self):
        print 'Number of original json objs is ', len(self.json_objs_list)
        json_objs_map = {}
        for curr_json_obj in self.json_objs_list:
            curr_tuple_key = []
            #
            assert len(curr_json_obj[cdjf.evidence]) == 1
            curr_evidence = curr_json_obj[cdjf.evidence][0]
            curr_tuple_key.append(curr_evidence)
            #
            if curr_json_obj[cdjf.extracted_information][cdjf.interaction_type] in [dit.adds_modification, dit.removes_modification]:
                curr_interaction_type = curr_json_obj[cdjf.extracted_information][cdjf.modification][0][cdjf.modification_type]
            else:
                curr_interaction_type = curr_json_obj[cdjf.extracted_information][cdjf.interaction_type]
            curr_tuple_key.append(curr_interaction_type)
            #
            participant_a = None
            if curr_json_obj[cdjf.extracted_information][cdjf.participant_a] is not None:
                participant_a = curr_json_obj[cdjf.extracted_information][cdjf.participant_a][cdjf.entity_text]
            curr_tuple_key.append(participant_a)
            #
            if cdjf.entities in curr_json_obj[cdjf.extracted_information][cdjf.participant_b]:
                assert curr_interaction_type in [dit.increases, dit.decreases]
                assert len(curr_json_obj[cdjf.extracted_information][cdjf.participant_b][cdjf.entities]) == 2
                participant_b0 = curr_json_obj[cdjf.extracted_information][cdjf.participant_b][cdjf.entities][0][cdjf.entity_text]
                participant_b1 = curr_json_obj[cdjf.extracted_information][cdjf.participant_b][cdjf.entities][1][cdjf.entity_text]
                curr_tuple_key.append(participant_b0)
                curr_tuple_key.append(participant_b1)
            else:
                participant_b = curr_json_obj[cdjf.extracted_information][cdjf.participant_b][cdjf.entity_text]
                curr_tuple_key.append(participant_b)
            #
            curr_tuple_key = tuple(curr_tuple_key)
            #
            if curr_tuple_key not in json_objs_map:
                json_objs_map[curr_tuple_key] = curr_json_obj
            else:
                print 'duplicate is ', json.dumps(curr_json_obj, ensure_ascii=True, sort_keys=True, indent=5)
                if json_objs_map[curr_tuple_key][cdjf.weight] < curr_json_obj[cdjf.weight]:
                    json_objs_map[curr_tuple_key] = curr_json_obj
            #
        self.json_objs_list = json_objs_map.values()
        print 'Number of json objects after filtering is ', len(self.json_objs_list)
        return self.json_objs_list

    def swap_catalyst_protein(self):
        cps_obj = cps.CatalystProteinStats()
        for curr_json_obj in self.json_objs_list:
            if curr_json_obj[cdjf.weight] > 0.2:
                continue
            if curr_json_obj[cdjf.extracted_information][cdjf.participant_a] is None:
                continue
            if cdjf.entities in curr_json_obj[cdjf.extracted_information][cdjf.participant_b]:
                continue
            #
            participant_a = curr_json_obj[cdjf.extracted_information][cdjf.participant_a][cdjf.entity_text]
            if participant_a in cps_obj.catalyst_prob_map:
                participant_a_catalyst_prob = cps_obj.catalyst_prob_map[participant_a]
            else:
                continue
            #
            participant_b = curr_json_obj[cdjf.extracted_information][cdjf.participant_b][cdjf.entity_text]
            if participant_b in cps_obj.catalyst_prob_map:
                participant_b_catalyst_prob = cps_obj.catalyst_prob_map[participant_b]
            else:
                continue
            #
            if (participant_b_catalyst_prob-participant_a_catalyst_prob) < 0.05:
                continue
            #
            print 'swapping catalyst protein for ', json.dumps(curr_json_obj, ensure_ascii=True, sort_keys=True, indent=5)
            temp = curr_json_obj[cdjf.extracted_information][cdjf.participant_a]
            curr_json_obj[cdjf.extracted_information][cdjf.participant_a] =\
                curr_json_obj[cdjf.extracted_information][cdjf.participant_b]
            curr_json_obj[cdjf.extracted_information][cdjf.participant_b] = temp
            print 'after swapping, ', json.dumps(curr_json_obj, ensure_ascii=True, sort_keys=True, indent=5)
        return self.json_objs_list

    def filter_top_fr_interaction_type_in_amr(self):
        print 'Number of json objects before top filtering is ', len(self.json_objs_list)
        max_num = 10
        weight_map = {}
        for curr_json_obj in self.json_objs_list:
            curr_weight = curr_json_obj[cdjf.weight]
            #
            if curr_json_obj[cdjf.extracted_information][cdjf.interaction_type] in [dit.adds_modification, dit.removes_modification]:
                curr_interaction_type = curr_json_obj[cdjf.extracted_information][cdjf.modification][0][cdjf.modification_type]
            else:
                curr_interaction_type = curr_json_obj[cdjf.extracted_information][cdjf.interaction_type]
            #
            assert len(curr_json_obj[cdjf.evidence]) == 1
            curr_evidence = curr_json_obj[cdjf.evidence][0]
            #
            curr_key = (curr_evidence, curr_interaction_type)
            #
            if curr_key not in weight_map:
                weight_map[curr_key] = {}
            #
            if curr_weight not in weight_map[curr_key]:
                weight_map[curr_key][curr_weight] = []
            weight_map[curr_key][curr_weight].append(curr_json_obj)
        #
        new_json_objs_list = []
        for curr_key in weight_map:
            print curr_key
            weight_keys = weight_map[curr_key].keys()
            weight_keys.sort(reverse=True)
            print weight_keys
            curr_new_json_objs_list = []
            for i in range(min(max_num, len(weight_keys))):
                curr_high_weight = weight_keys[i]
                print curr_high_weight
                curr_new_json_objs_list += weight_map[curr_key][curr_high_weight]
            print '******************'
            for curr_item in curr_new_json_objs_list:
                print 'curr_item is ', json.dumps(curr_item, ensure_ascii=True, sort_keys=True, indent=5)
            print '******************'
            new_json_objs_list += curr_new_json_objs_list
        self.json_objs_list = new_json_objs_list
        print 'Number of json objects after top filtering is ', len(self.json_objs_list)
        return self.json_objs_list








