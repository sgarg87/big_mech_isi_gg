import csv
import json
import constants_darpa_json_format as cdjf
import glob


global_path = '../../../protein_protein_interaction_datasets/'


def get_hprd_data():
    list_of_protein_protein_tuples = []
    with open(global_path+'HPRD_Release9_062910/BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt', 'rU') as f:
        reader = csv.reader(f, delimiter='\t')
        for curr_row in reader:
            print curr_row[0], curr_row[3]
            curr_tuple = tuple([curr_row[0], curr_row[3]])
            list_of_protein_protein_tuples.append(curr_tuple)
    return list_of_protein_protein_tuples


def get_mint_data():
    list_of_protein_protein_tuples = []
    with open(global_path+'2012-10-29-mint-full.txt', 'rU') as f:
        reader = csv.reader(f, delimiter='\t')
        for curr_row in reader:
            # print curr_row
            print curr_row[4], curr_row[5]
            curr_tuple = tuple([curr_row[4], curr_row[5]])
            list_of_protein_protein_tuples.append(curr_tuple)
    return list_of_protein_protein_tuples


def get_aimed_data():
    list_of_protein_protein_tuples = []
    with open(global_path+'aimed_bioc_relations.csv', 'r') as f:
        reader = csv.DictReader(f)
        for curr_row in reader:
            print curr_row['Arg1'], curr_row['Arg2']
            curr_tuple = tuple([curr_row['Arg1'], curr_row['Arg2']])
            list_of_protein_protein_tuples.append(curr_tuple)
    return list_of_protein_protein_tuples


def get_chicago_data():
    list_of_protein_protein_tuples = []
    with open(global_path+'stats_dataout.json', 'r') as f:
        map_chicago = json.load(f)
        for curr_list in map_chicago.values():
            for curr_tuple in curr_list:
                if curr_tuple[1] == 1:
                    proteins_list_tuple = tuple(curr_tuple[0][1:])
                    print proteins_list_tuple
                    list_of_protein_protein_tuples.append(proteins_list_tuple)
    return list_of_protein_protein_tuples


def get_pubmed45_data():
    list_of_protein_protein_tuples = []
    with open(global_path+'pub_med_45_interactions.json', 'r') as f:
        pubmed_list = json.load(f)
        for curr_tuple in pubmed_list:
            proteins_list_tuple = curr_tuple[1:]
            if len(proteins_list_tuple) == 2:
                if None in proteins_list_tuple:
                    continue
                else:
                    list_of_protein_protein_tuples.append(tuple(proteins_list_tuple))
            elif len(proteins_list_tuple) == 3:
                list_of_protein_protein_tuples.append(tuple(proteins_list_tuple[1:]))
            else:
                raise AssertionError
    return list_of_protein_protein_tuples


def get_model_data():
    list_of_protein_protein_tuples = []
    #
    list_json_files = glob.glob(global_path+"biopax_model/*.json")
    print list_json_files
    for curr_json_file in list_json_files:
        curr_tuple = []
        with open(curr_json_file, 'r') as f:
            curr_json_map = json.load(f)
            extracted_information = curr_json_map[cdjf.extracted_information]
            curr_json_map = None
            participant_a = extracted_information[cdjf.participant_a]
            participant_b = extracted_information[cdjf.participant_b]
            if participant_a is None:
                continue
            #
            # print 'participant_b', participant_b
            if cdjf.entities in participant_b:
                for curr_entity in participant_b[cdjf.entities]:
                    curr_tuple.append(curr_entity[cdjf.entity_text])
            else:
                curr_tuple.append(participant_a[cdjf.entity_text])
                curr_tuple.append(participant_b[cdjf.entity_text])
            #
            curr_tuple = tuple(curr_tuple)
            print curr_tuple
        #
        list_of_protein_protein_tuples.append(curr_tuple)
    return list_of_protein_protein_tuples


if __name__ == '__main__':
    list_of_protein_protein_tuples = get_hprd_data()
    list_of_protein_protein_tuples += get_mint_data()
    list_of_protein_protein_tuples += get_aimed_data()
    list_of_protein_protein_tuples += get_chicago_data()
    list_of_protein_protein_tuples += get_pubmed45_data()
    list_of_protein_protein_tuples += get_model_data()
    print len(list_of_protein_protein_tuples)
    #
    with open(global_path+'./list_of_protein_protein_tuples.json', 'w') as f:
        json.dump(list_of_protein_protein_tuples, f, indent=4)


