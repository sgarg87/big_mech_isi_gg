import xml.etree.ElementTree as et
import csv
import json

path = './AImed/aimed_bioc'


def get_all_sentences_and_proteins():
    et_obj = et.parse(path+'.xml').getroot()
    sentence_id__proteins_list_map = {}
    f_t = open(path+'_sentences.txt', 'w')
    json_obj = {}
    for curr_document in et_obj.findall('document'):
        curr_sentence_id = curr_document.findall('id')
        assert len(curr_sentence_id) == 1
        curr_sentence_id = curr_sentence_id[0].text
        #
        json_obj[curr_sentence_id] = {}
        #
        curr_passages_text = curr_document.findall('passage/text')
        assert len(curr_passages_text) == 1
        curr_passages_text = curr_passages_text[0].text
        num_sentences = curr_passages_text.count('\n')
        #
        json_obj[curr_sentence_id]['num_sentences'] = num_sentences
        #
        curr_sentence_text = curr_passages_text.replace('\n', ' ')
        #
        json_obj[curr_sentence_id]['text'] = curr_sentence_text
        #
        f_t.write(curr_sentence_id+'\t'+curr_sentence_text)
        f_t.write('\n')
        #
        curr_proteins_list = []
        for curr_annotation in curr_document.findall('passage/annotation'):
            infons = curr_annotation.findall('infon')
            is_protein = False
            for curr_infon in infons:
                curr_infon_key = curr_infon.get('key')
                if curr_infon_key == 'type':
                    assert curr_infon.text == 'protein'
                    is_protein = True
            assert is_protein
            curr_annotation_text = curr_annotation.findall('text')
            assert len(curr_annotation_text) == 1
            curr_annotation_text = curr_annotation_text[0]
            curr_protein_text = curr_annotation_text.text
            curr_proteins_list.append(curr_protein_text)
        #
        json_obj[curr_sentence_id]['proteins_list'] = curr_proteins_list
        sentence_id__proteins_list_map[curr_sentence_id] = curr_proteins_list
        #
    f_t.close()
    #
    with open(path+'_proteins.txt', 'w') as f:
        for curr_sentence_id in set(sentence_id__proteins_list_map):
            curr_proteins_list = sentence_id__proteins_list_map[curr_sentence_id]
            for curr_protein in curr_proteins_list:
                f.write(curr_sentence_id+'\t'+curr_protein)
                f.write('\n')
    #
    with open(path+'_sentences_proteins.json', 'w') as f:
        json.dump(json_obj, f, indent=4)


def get_all_relations():
    et_obj = et.parse(path+'.xml').getroot()
    relations = []
    #
    json_obj = {}
    #
    for curr_document in et_obj.findall('document'):
        curr_sentence_id = curr_document.findall('id')
        assert len(curr_sentence_id) == 1
        curr_sentence_id = curr_sentence_id[0].text
        #
        json_obj[curr_sentence_id] = {}
        json_obj[curr_sentence_id]['relations'] = []
        #
        curr_passage = curr_document.findall('passage')
        assert len(curr_passage) == 1
        curr_passage = curr_passage[0]
        curr_proteins_map = {}
        for curr_annotation in curr_passage.findall('annotation'):
            curr_protein_id = curr_annotation.get('id')
            infons = curr_annotation.findall('infon')
            for curr_infon in infons:
                curr_infon_key = curr_infon.get('key')
                if curr_infon_key == 'type':
                    assert curr_infon.text == 'protein'
            curr_annotation_text = curr_annotation.findall('text')
            assert len(curr_annotation_text) == 1
            curr_annotation_text = curr_annotation_text[0]
            curr_protein_text = curr_annotation_text.text
            curr_proteins_map[curr_protein_id] = curr_protein_text
        #
        for curr_relation in curr_passage.findall('relation'):
            curr_relation_map = {}
            curr_relation_map['sentence_id'] = curr_sentence_id
            nodes = curr_relation.findall('node')
            for curr_node in nodes:
                curr_protein_id = curr_node.get('refid')
                curr_protein_name = curr_proteins_map[curr_protein_id]
                curr_role = curr_node.get('role')
                assert curr_role not in curr_relation_map
                curr_relation_map[curr_role] = curr_protein_name
            #
            relations.append(curr_relation_map)
            #
            json_obj[curr_sentence_id]['relations'].append(curr_relation_map)
            #
            print curr_relation_map
    #
    with open(path+'_sentence_relations.json', 'w') as f:
        json.dump(json_obj, f, indent=4)
    #
    with open(path+'_relations.csv', 'w') as f:
        field_names = ['sentence_id', 'Arg1', 'Arg2']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for curr_relation_map in relations:
            writer.writerow(curr_relation_map)


if __name__ == '__main__':
    get_all_sentences_and_proteins()
    get_all_relations()

