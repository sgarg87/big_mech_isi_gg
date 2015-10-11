from constants import *
import copy
from config import *
from config_console_output import *


is_compress = False

# todo: define a base class and then have these classes inherited since there are some common attributes and come functionality is also in common

class ComplexTypeInteraction:
    def __init__(self, protein_1_str, protein_1_state_str_list, protein_2_str, protein_2_state_str_list, catalyst_str, catalyst_state_str_list, complex_str, complex_state_str_list, is_positive_catalyst=True, weight=1, complex_interaction_str=None):
        if protein_1_str is not None:
            protein_1_str = protein_1_str.upper()
        self.protein_1_str = protein_1_str
        if is_compress:
            self.protein_1_state_str_list = compress_state_name(protein_1_state_str_list)
        else:
            self.protein_1_state_str_list = protein_1_state_str_list
        if protein_2_str is not None:
            protein_2_str = protein_2_str.upper()
        self.protein_2_str = protein_2_str
        if is_compress:
            self.protein_2_state_str_list = compress_state_name(protein_2_state_str_list)
        else:
            self.protein_2_state_str_list = protein_2_state_str_list
        if catalyst_str is not None:
            catalyst_str = catalyst_str.upper()
        self.catalyst_str = catalyst_str
        if is_compress:
            self.catalyst_state_str_list = compress_state_name(catalyst_state_str_list)
        else:
            self.catalyst_state_str_list = catalyst_state_str_list
        if complex_str is not None:
            complex_str = complex_str.upper()
        self.complex_str = complex_str
        if is_compress:
            self.complex_state_str_list = compress_state_name(complex_state_str_list)
        else:
            self.complex_state_str_list = complex_state_str_list
        self.is_positive_catalyst = is_positive_catalyst
        self.weight = weight
        self.complex_interaction_str = complex_interaction_str
        self.text_sentence = None
        #
        self.is_left_to_right = True
        #
        self.protein_1_id = None
        self.protein_2_id = None
        self.complex_id = None
        self.catalyst_id = None
        #
        self.protein_1_id_prob = None
        self.protein_2_id_prob = None
        self.complex_id_prob = None
        self.catalyst_id_prob = None
        #
        self.protein_1_type = None
        self.protein_2_type = None
        self.complex_type = None
        self.catalyst_type = None
        #
        self.is_negative_information = False
        self.other_type = None
        self.other_source_id = None
        #
        self.pmc_id = None
        #
        self.model_relation_map = {}
        self.protein_comma_list_link_to_model = None

    def __eq__(self, other):
        self_tuple = (self.catalyst_str, (set(self.catalyst_state_str_list) if self.catalyst_state_str_list is not None else None), self.is_positive_catalyst, self.protein_1_str, set(self.protein_1_state_str_list), self.protein_2_str, set(self.protein_2_state_str_list), (set(self.complex_state_str_list) if self.complex_state_str_list is not None else None))
        other_tuple = (other.catalyst_str, (set(other.catalyst_state_str_list) if other.catalyst_state_str_list is not None else None), other.is_positive_catalyst, other.protein_1_str, set(other.protein_1_state_str_list), other.protein_2_str, set(other.protein_2_state_str_list), (set(other.complex_state_str_list) if other.complex_state_str_list is not None else None))
        if debug:
            print 'self_tuple:', self_tuple
            print 'other_tuple:', other_tuple
        if self_tuple == other_tuple:
            return True
        self_tuple = (self.catalyst_str, set(self.catalyst_state_str_list), self.is_positive_catalyst, self.protein_2_str, set(self.protein_2_state_str_list), self.protein_1_str, set(self.protein_1_state_str_list), set(self.complex_state_str_list))
        if debug:
            print 'self_tuple:', self_tuple
        if self_tuple == other_tuple:
            return True
        return False

    def conflict(self, other):
        other_conflict = copy.deepcopy(other)
        other_conflict.is_positive_catalyst = not other.is_positive_catalyst
        return self == other_conflict

    def __str__(self):
        complex_str = 'complex_type_interaction:{\n\t'
        if self.other_type is not None and self.other_type:
            complex_str += '\n\tother_type: '+self.other_type + '\n\t'
        if self.text_sentence is not None and self.text_sentence:
            complex_str += '\n\tsentence: ' + self.text_sentence + '\n\t'
        if self.other_source_id is not None and self.other_source_id:
            complex_str += '\n\tother_source_id: ' + self.other_source_id + '\n\t'
        if self.weight is not None:
            complex_str += 'weight:'
            complex_str += str(round(self.weight, weight_round_decimal))
            complex_str += '\n\t'
        if self.catalyst_str is not None:
            if self.is_positive_catalyst:
                complex_str += 'catalyst:' + self.catalyst_str
            else:
                complex_str += 'inhibitor:' + self.catalyst_str
            if (self.catalyst_state_str_list is not None) and self.catalyst_state_str_list:
                complex_str += get_state_list_str(self.catalyst_state_str_list)
        complex_str += '\n\t'
        complex_str += self.protein_1_str
        if (self.protein_1_state_str_list is not None) and self.protein_1_state_str_list:
            complex_str += get_state_list_str(self.protein_1_state_str_list)
        if self.complex_interaction_str is None or not self.complex_interaction_str:
            complex_str += ' bind '
        else:
            complex_str += ' ' + self.complex_interaction_str + ' '
        complex_str += self.protein_2_str
        if (self.protein_2_state_str_list is not None) and self.protein_2_state_str_list:
            complex_str += get_state_list_str(self.protein_2_state_str_list)
        if self.complex_str is not None and self.complex_str:
            complex_str += '  ---->  '
            complex_str += self.complex_str
            if (self.complex_state_str_list is not None) and self.complex_state_str_list:
                complex_str += get_state_list_str(self.complex_state_str_list)
        complex_str += '\n\t}'
        return complex_str

    def english(self):
        complex_str = ''
        if self.catalyst_str is not None:
            if self.is_positive_catalyst:
                complex_str += 'With catalyst ' + self.catalyst_str
            else:
                complex_str += 'With inhibitor ' + self.catalyst_str
            if (self.catalyst_state_str_list is not None) and self.catalyst_state_str_list:
                complex_str += ' in state '
                complex_str += get_state_list_str(self.catalyst_state_str_list)
            complex_str += ', '
        complex_str += self.protein_1_str
        if (self.protein_1_state_str_list is not None) and self.protein_1_state_str_list:
            complex_str += ' in state '
            complex_str += get_state_list_str(self.protein_1_state_str_list)
        if self.complex_interaction_str is None or not self.complex_interaction_str:
            complex_str += ' binds with '
        else:
            complex_str += ' ' + self.complex_interaction_str + ' '
        complex_str += self.protein_2_str
        if (self.protein_2_state_str_list is not None) and self.protein_2_state_str_list:
            complex_str += ' in state '
            complex_str += get_state_list_str(self.protein_2_state_str_list)
        if self.complex_str is not None and self.complex_str:
            complex_str += '  to form '
            complex_str += self.complex_str
            if (self.complex_state_str_list is not None) and self.complex_state_str_list:
                complex_str += ' in state '
                complex_str += get_state_list_str(self.complex_state_str_list)
        return complex_str

    def contains_nonentity(self):
        general_entity_names = protein_labels+protein_part_labels
        general_entity_names = [x.upper() for x in general_entity_names]
        if self.protein_1_str in general_entity_names:
            return True
        elif self.protein_2_str in general_entity_names:
            return True
        elif self.catalyst_str in general_entity_names:
            return True
        elif self.complex_str in general_entity_names:
            return True
        return False


class Interaction:
    def __init__(self, protein_str, protein_state_str_list, catalyst_str, catalyst_state_str_list, result_state_str_list, interaction_str, is_positive_catalyst=True, weight=1):
        #interaction_str is redundant for understanding of a given interaction, it represent type of interaction
        #_str means string format
        if protein_str is not None:
            protein_str = protein_str.upper()
        self.protein_str = protein_str
        if is_compress:
            self.protein_state_str_list = compress_state_name(protein_state_str_list)
        else:
            self.protein_state_str_list = protein_state_str_list
        if catalyst_str is not None:
            catalyst_str = catalyst_str.upper()
        self.catalyst_str = catalyst_str
        if is_compress:
            self.catalyst_state_str_list = compress_state_name(catalyst_state_str_list)
        else:
            self.catalyst_state_str_list = catalyst_state_str_list
        if is_compress:
            self.result_state_str_list = compress_state_name(result_state_str_list)
        else:
            self.result_state_str_list = result_state_str_list
        self.interaction_str = interaction_str
        self.is_positive_catalyst = is_positive_catalyst
        if is_compress:
            if self.interaction_str == dephosphorylate:
                if phosphorylate not in self.protein_state_str_list:
                    self.protein_state_str_list.append(phosphorylate)
                if phosphorylate in self.result_state_str_list:
                    self.result_state_str_list.remove(phosphorylate)
        self.weight = weight
        self.text_sentence = None
        #
        self.protein_id = None
        self.catalyst_id = None
        #
        self.protein_id_prob = None
        self.catalyst_id_prob = None
        #
        self.protein_type = None
        self.catalyst_type = None
        #
        self.is_negative_information = False
        #
        self.protein_result_str = None
        self.protein_result_id = None
        self.protein_result_type = None
        #
        self.other_type = None
        self.other_source_id = None
        #
        self.pmc_id = None
        #
        self.model_relation_map = {}
        self.protein_comma_list_link_to_model = None


    def __eq__(self, other):
        self_tuple = (self.catalyst_str, set(self.catalyst_state_str_list), self.is_positive_catalyst, self.protein_str, set(self.protein_state_str_list), set(self.result_state_str_list))
        other_tuple = (other.catalyst_str, set(other.catalyst_state_str_list), other.is_positive_catalyst, other.protein_str, set(other.protein_state_str_list), set(other.result_state_str_list))
        if debug:
            print 'self_tuple:', self_tuple
            print 'other_tuple:', other_tuple
        if self_tuple == other_tuple:
            return True
        return False

    def __str__(self):
        interaction_str = 'state_change:{\n\t'
        if self.other_type is not None and self.other_type:
            interaction_str += '\n\tother_type: ' + self.other_type + '\n\t'
        if self.text_sentence is not None and self.text_sentence:
            interaction_str += '\n\tsentence: ' + self.text_sentence + '\n\t'
        if self.other_source_id is not None and self.other_source_id:
            interaction_str += '\n\tother_source_id: ' + self.other_source_id + '\n\t'
        if self.weight is not None:
            interaction_str += 'weight:'
            interaction_str += str(round(self.weight, weight_round_decimal))
            interaction_str += '\n\t'
        if self.catalyst_str is not None:
            interaction_str += ('catalyst:' if self.is_positive_catalyst else 'inhibitor:') + self.catalyst_str
            if (self.catalyst_state_str_list is not None) and self.catalyst_state_str_list:
                interaction_str += get_state_list_str(self.catalyst_state_str_list)
        interaction_str += '\n\t'
        interaction_str += self.protein_str
        if (self.protein_state_str_list is not None) and self.protein_state_str_list:
            interaction_str += get_state_list_str(self.protein_state_str_list)
        interaction_str += '  ---->  '
        interaction_str += self.protein_str
        if (self.result_state_str_list is not None) and self.result_state_str_list:
            interaction_str += get_state_list_str(self.result_state_str_list)
        interaction_str += '\n\t}'
        return interaction_str

    def english(self):
        interaction_str = ''
        if self.catalyst_str is not None:
            interaction_str += self.catalyst_str
            if (self.catalyst_state_str_list is not None) and self.catalyst_state_str_list:
                interaction_str += ' with state '
                interaction_str += get_state_list_str(self.catalyst_state_str_list)
            if self.is_positive_catalyst:
                interaction_str += ' catalyses state change of '
            else:
                interaction_str += ' inhibits state change of '
            interaction_str += self.protein_str
            interaction_str += ' from '
            if self.protein_state_str_list is not None:
                interaction_str += get_state_list_str(self.protein_state_str_list)
            else:
                interaction_str += str([])
            interaction_str += ' to '
            if self.result_state_str_list is not None:
                interaction_str += get_state_list_str(self.result_state_str_list)
            else:
                interaction_str += str([])
        else:
            interaction_str += 'State of '
            interaction_str += self.protein_str
            interaction_str += ' is changed'
            interaction_str += ' from '
            if self.protein_state_str_list is not None:
                interaction_str += get_state_list_str(self.protein_state_str_list)
            else:
                interaction_str += str([])
            interaction_str += ' to '
            if self.result_state_str_list is not None:
                interaction_str += get_state_list_str(self.result_state_str_list)
            else:
                interaction_str += str([])
        return interaction_str

    def contains_nonentity(self):
        general_entity_names = protein_labels+protein_part_labels
        general_entity_names = [x.upper() for x in general_entity_names]
        if self.catalyst_str in general_entity_names:
            return True
        elif self.protein_str in general_entity_names:
            return True
        return False

    def is_mechanistic_information(self):
        if is_site_information_in_state_list(self.protein_state_str_list):
            return True
        elif is_site_information_in_state_list(self.result_state_str_list):
            return True
        else:
            return False


def compress_state_name(state_names_list):
    if state_names_list is None or not state_names_list:
        return []
    new_state_names_list = []
    no_state_flag = True
    for state_name in state_names_list:
        state_name = no_quotes_regexp.sub('', state_name)
        if state_name in state_labels or mutate in state_name:
            no_state_flag = False
        if state_name == hyperphosphorylate:
            state_name = phosphorylate
        elif state_name == dephosphorylate:
            state_name = None
        if state_name is not None:
            new_state_names_list.append(state_name)
    #if no adjective kind flag in state list, state list should be [].
    # It doesn't make sense to just have a site information say EFFR[T669]
    if no_state_flag:
        return []
    else:
        return list(set(new_state_names_list)) #unique elements in the list returned


def get_state_list_str(state_list):
    if state_list is None or not state_list:
        return str([])
    else:
        state_list_str = '['
        for state in state_list:
            state_list_str += state
            state_list_str += ','
        state_list_str = state_list_str.rstrip(',')
        state_list_str += ']'
        return state_list_str


def is_site_information_in_state_list(state_list):
    not_state_label_present = False
    for state in state_list:
        if state not in state_labels:
            return True
    return False
