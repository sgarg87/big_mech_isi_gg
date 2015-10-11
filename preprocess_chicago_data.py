import csv
import copy
import json


def get_label(is_correct):
    if is_correct == 'AM_R_CORRECT':
        curr_label = 1
    elif is_correct == 'AM_R_WRONG':
        curr_label = 0
    elif is_correct == 'AM_R_DONTKNOW':
        curr_label = 0
    else:
        print is_correct
        raise AssertionError
    return curr_label


def validate_interaction_tuple(tuple, row, label):
    is_valid = True
    if tuple[0] == 'express':
        if len(tuple) != 5:
            is_valid = False
    elif tuple[0] == 'colocalize':
        if len(tuple) != 5:
            is_valid = False
    if not is_valid:
        pass
        # print '************'
        # print row[-3]
        # print row[-2]
        # print row[-1]
        # print tuple
        # print label
    else:
        if len(tuple) in [4, 5]:
            print '+++++++++++'
            print row[-3]
            print row[-2]
            print row[-1]
            print tuple
            print label


def add_protein_to_map(proteins_list_map, id, protein):
    if id not in proteins_list_map:
        proteins_list_map[id] = []
    if protein not in proteins_list_map[id]:
        proteins_list_map[id].append(protein)


def read_data_file():
    interaction_tuples_map = {}
    proteins_list_map = {}
    interaction_types_list = []
    annotators_list = ['lata', 'aditi', 'monali', 'singh']
    with open('../chicago_data/stats_dataout.csv', 'r') as f:
        reader = csv.reader(f, delimiter=':')
        for curr_row in reader:
            curr_id = curr_row[0]
            curr_id = curr_id.split('\t')[0]
            curr_id = curr_id.strip()
            #
            curr_interaction_type = curr_row[2].strip()
            if curr_interaction_type not in interaction_types_list:
                interaction_types_list.append(curr_interaction_type)
            #
            curr_catalyst = curr_row[3].strip()
            curr_protein = curr_row[4].strip()
            curr_interaction_tuple = [curr_interaction_type, curr_catalyst, curr_protein]
            #
            curr_annotator = curr_row[5].strip().lower()
            #
            curr_protein2 = None
            curr_protein3 = None
            #
            if curr_annotator in annotators_list:
                is_correct = curr_row[6].strip()
            else:
                curr_protein2 = curr_row[5].strip()
                curr_annotator = curr_row[6].strip().lower()
                if curr_annotator in annotators_list:
                    is_correct = curr_row[7].strip()
                else:
                    curr_protein3 = curr_row[6].strip()
                    curr_annotator = curr_row[7].strip().lower()
                    assert curr_annotator in annotators_list
                    is_correct = curr_row[8].strip()
            curr_label = get_label(is_correct)
            #
            if curr_protein2 is not None:
                curr_interaction_tuple.append(curr_protein2)
            if curr_protein3 is not None:
                curr_interaction_tuple.append(curr_protein3)
            #
            curr_interaction_tuple = tuple(curr_interaction_tuple)
            if len(curr_interaction_tuple) > 3:
                print '+'
                continue
            #
            add_protein_to_map(proteins_list_map, curr_id, curr_catalyst)
            add_protein_to_map(proteins_list_map, curr_id, curr_protein)
            #
            # validate_interaction_tuple(curr_interaction_tuple, curr_row, curr_label)
            #
            # if len(curr_interaction_tuple) == 5:
            #     print '*******************************************'
            #     print curr_row[-3]
            #     print curr_row[-2]
            #     print curr_row[-1]
            #     print '{}, {}'.format(curr_interaction_tuple, curr_label)
            #
            curr_tuple = tuple([curr_interaction_tuple, curr_label])
            opp_tuple = tuple([curr_interaction_tuple, int(not curr_label)])
            #
            curr_sentence = curr_row[-1].strip()
            if not curr_sentence:
                # print curr_row
                curr_sentence = curr_row[-2].strip()
                if not curr_sentence:
                    continue
            #
            if curr_id not in interaction_tuples_map:
                interaction_tuples_map[curr_id] = []
            #
            if curr_label == 0:
                if opp_tuple in interaction_tuples_map[curr_id]:
                    continue
                elif curr_tuple in interaction_tuples_map[curr_id]:
                    continue
                else:
                    interaction_tuples_map[curr_id].append(curr_tuple)
            elif curr_label == 1:
                if curr_tuple in interaction_tuples_map[curr_id]:
                    continue
                elif opp_tuple in interaction_tuples_map[curr_id]:
                    interaction_tuples_map[curr_id].remove(opp_tuple)
                    interaction_tuples_map[curr_id].append(curr_tuple)
                else:
                    interaction_tuples_map[curr_id].append(curr_tuple)
            else:
                raise AssertionError
    #
    json_f = open('../chicago_data/stats_dataout.json', 'w')
    json.dump(interaction_tuples_map, json_f, indent=4)
    json_f.close()
    #
    json_f = open('../chicago_data/stats_proteins.json', 'w')
    json.dump(proteins_list_map, json_f, indent=4)
    json_f.close()
    #
    json_f = open('../chicago_data/stats_interaction_types.json', 'w')
    json.dump(interaction_types_list, json_f, indent=4)
    json_f.close()


if __name__ == '__main__':
    read_data_file()
