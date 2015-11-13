import constants_darpa_json_format as cdjf
import darpa_interaction_types as dit
from config_reset_console_output import *
import catalyst_protein_stats as cps
import json


class PostProcessJSONDARPA:
    def __init__(self, json_objs_list):
        self.json_objs_list = json_objs_list

    def remove_identifier_prob_list(self):
        for curr_json_obj in self.json_objs_list:
            if cdjf.identifier_prob_list in curr_json_obj:
                curr_json_obj.pop(cdjf.identifier_prob_list, None)
        #
        # return self.json_objs_list

    def match_symmetric_duplicates(self, tuple, tuples_list):
        if tuple in tuples_list:
            return True, tuple
        else:
            evidence = tuple[0]
            interaction_type = tuple[1]
            participant_a = tuple[2]
            if interaction_type not in [dit.increases, dit.decreases, dit.binds]:
                return False, None
            elif participant_a is None:
                return False, None
            elif len(tuple) == 4 and interaction_type in [dit.increases, dit.decreases]:
                return False, None
            else:
                participant_b = tuple[3]
                if len(tuple) == 5:
                    participant_b2 = tuple[4]
                #
                for curr_tuple_in_list in tuples_list:
                    if curr_tuple_in_list[0] != evidence:
                        continue
                    elif curr_tuple_in_list[1] != interaction_type:
                        continue
                    elif len(curr_tuple_in_list) != len(tuple):
                        continue
                    elif len(tuple) == 4 and curr_tuple_in_list[2] == participant_a and curr_tuple_in_list[3] == participant_b:
                        raise AssertionError, 'This is a perfect match. It should have been identified above in the beginning steps itself.'
                    elif len(tuple) == 4 and curr_tuple_in_list[3] == participant_a and curr_tuple_in_list[2] == participant_b:
                        return True, curr_tuple_in_list
                    elif len(tuple) == 5 and curr_tuple_in_list[3] == participant_b2 and curr_tuple_in_list[4] == participant_b:
                        return True, curr_tuple_in_list
                    else:
                        continue
                return False, None

    def filter_out_duplicates_fr_evidence(self, is_set=False):
        def find_match_key(key, json_objs_map):
            if not is_set:
                if key in json_objs_map:
                    return key
                else:
                    return None
            else:
                match_key_list = []
                for curr_key in json_objs_map:
                    if set(key) == set(curr_key):
                        match_key_list.append(curr_key)
                #
                assert len(match_key_list) in [0, 1]
                #
                if match_key_list:
                    return match_key_list[0]

        # is_set True means that the most likely of all interactions involving same set of proteins and interaction is chosen
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
            # differentiation between set and tuple is crucial
            # both play different role for identifying duplicates
            # set corresponds to all interactions with same of proteins and interactions being duplicates to each other
            # whereas tuple does not
            #
            curr_tuple_key = tuple(curr_tuple_key)
            #
            curr_match_tuple_key = find_match_key(curr_tuple_key, json_objs_map)
            if curr_match_tuple_key is None:
                json_objs_map[curr_tuple_key] = curr_json_obj
            else:
                print 'duplicate is ', json.dumps(curr_json_obj, ensure_ascii=True, sort_keys=True, indent=4)
                if json_objs_map[curr_match_tuple_key][cdjf.weight] < curr_json_obj[cdjf.weight]:
                    json_objs_map[curr_tuple_key] = curr_json_obj
                    json_objs_map.pop(curr_match_tuple_key, None)
            #
        self.json_objs_list = json_objs_map.values()
        print 'Number of json objects after filtering is ', len(self.json_objs_list)

    def filter_out_symmetric_duplicates_fr_evidence(self):
        # using set, the code may be simplified
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
            is_match_prv, match_tuple = self.match_symmetric_duplicates(curr_tuple_key, json_objs_map.keys())
            if not is_match_prv:
                json_objs_map[curr_tuple_key] = curr_json_obj
            else:
                print 'symmetric duplicate is ', json.dumps(curr_json_obj, ensure_ascii=True, sort_keys=True, indent=5)
                if json_objs_map[match_tuple][cdjf.weight] < curr_json_obj[cdjf.weight]:
                    json_objs_map[curr_tuple_key] = curr_json_obj
                    json_objs_map.pop(match_tuple, None)
            #
        self.json_objs_list = json_objs_map.values()
        print 'Number of json objects after filtering is ', len(self.json_objs_list)

    def filter_out_increase_decrease_wd_no_catalyst(self):
        print 'Number of original json objs is ', len(self.json_objs_list)
        new_json_objs_list = []
        #
        for curr_json_obj in self.json_objs_list:
            if curr_json_obj[cdjf.extracted_information][cdjf.interaction_type] in [dit.adds_modification, dit.removes_modification]:
                new_json_objs_list.append(curr_json_obj)
            else:
                curr_interaction_type = curr_json_obj[cdjf.extracted_information][cdjf.interaction_type]
                if curr_interaction_type not in [dit.increases, dit.decreases]:
                    new_json_objs_list.append(curr_json_obj)
                else:
                    if curr_json_obj[cdjf.extracted_information][cdjf.participant_a] is not None:
                        new_json_objs_list.append(curr_json_obj)
                    else:
                        assert cdjf.entities not in curr_json_obj[cdjf.extracted_information][cdjf.participant_b]
        #
        self.json_objs_list = new_json_objs_list
        print 'Number of json objects after filtering is ', len(self.json_objs_list)
        # return self.json_objs_list

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
        # return self.json_objs_list

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
        # return self.json_objs_list








