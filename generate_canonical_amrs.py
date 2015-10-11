import constants as c
import model_json_interactions_to_amrs as ija
import os
import shutil
import constants_absolute_path as cap
import extract_from_amr_dot as ead
import gen_extractor_features_data as gefd
from config_console_output import *


class CanonicalAMRs:
    def __init__(self):
        #
        self.file_path = './canonical_amrs/'
        if os.path.exists(cap.absolute_path+self.file_path):
            shutil.rmtree(cap.absolute_path+self.file_path)
        os.makedirs(cap.absolute_path+self.file_path)
        #
        self.data = {}
        self.data[gefd.const_paths_map] = {}
        self.data[gefd.const_interaction_tuples_map] = {}
        self.data[gefd.const_sentences_map] = {}
        self.data[gefd.const_joint_labels_map] = {}
        #
        self.ija_obj = ija.InteractionsJSONToAMR()
        #
        # modifications
        self.concept_terms = []
        #
        self.modification_terms = [c.phosphorylate, c.acetylate, c.deacetylate, c.farnesylate, c.glycosylate, c.hydroxylate, c.methylate,
                         c.ribosylate, c.sumoylate, c.ubiquitinate, c.hydrolyze]
        self.concept_terms += self.modification_terms
        # increases activity
        self.activate_terms = [c.activate]
        self.concept_terms += self.activate_terms
        # activity
        self.activity_terms = [c.signal, c.act]
        self.concept_terms += self.activity_terms
        # decreases activity
        self.decrease_activity_terms = [c.impede, c.diminish, c.inhibit]
        self.concept_terms += self.decrease_activity_terms
        # translocate
        self.translocate_terms = [c.recruit, c.translocate, c.localize]
        self.concept_terms += self.translocate_terms
        # increases
        self.increase_terms = [c.increase, c.express, c.transcribe, c.produce]
        self.concept_terms += self.increase_terms
        #decrease
        self.decrease_terms = [c.decrease, c.degrade]
        self.concept_terms += self.decrease_terms
        # binds
        self.bind_terms = [c.bind, c.associate, c.heterodimerize, c.dissociate]
        self.concept_terms += self.bind_terms
        #
        self.pending_concept_terms = [c.form]
        #
        self.count = 0

    def add_interaction(self, curr_concept_term, catalyst_term, protein_term, protein2_term, label=1):
        #
        curr_amr_info_map = self.ija_obj.gen_amr_triplet_joint(curr_concept_term, catalyst_term, protein_term, protein2_term, label=label)
        #
        sentence = 'no text, label={}'.format(label)
        self.count += 1
        path_key = self.file_path+'interaction_'+str(self.count)
        ead.nodes_to_dot(curr_amr_info_map['nodes'], path_key, sentence)
        #
        self.data[gefd.const_paths_map][path_key] = curr_amr_info_map['nodes']
        self.data[gefd.const_interaction_tuples_map][path_key] = curr_amr_info_map['tuple']
        self.data[gefd.const_sentences_map][path_key] = sentence
        self.data[gefd.const_joint_labels_map][path_key] = label

    def gen_amrs(self, is_pickle=False):
        for curr_concept_term in self.concept_terms:
            for curr_protein_name_prefix in ['A', 'B', 'C', 'D', 'E']:
                if curr_concept_term not in self.bind_terms:
                    self.add_interaction(curr_concept_term, None, curr_protein_name_prefix+'1', None, label=1)
                    self.add_interaction(curr_concept_term, None, curr_protein_name_prefix+'1', None, label=0)
                    if curr_concept_term not in self.activity_terms:
                        self.add_interaction(curr_concept_term, curr_protein_name_prefix+'0', curr_protein_name_prefix+'1',
                                             None, label=1)
                        self.add_interaction(curr_concept_term, curr_protein_name_prefix+'0', curr_protein_name_prefix+'1',
                                             None, label=0)
                        self.add_interaction(curr_concept_term, curr_protein_name_prefix+'0', curr_protein_name_prefix+'1',
                                             None, label=2)
                else:
                    self.add_interaction(curr_concept_term, curr_protein_name_prefix+'0', curr_protein_name_prefix+'1',
                                         curr_protein_name_prefix+'2', label=1)
                    self.add_interaction(curr_concept_term, curr_protein_name_prefix+'0', curr_protein_name_prefix+'1',
                                         curr_protein_name_prefix+'2', label=0)
                    self.add_interaction(curr_concept_term, curr_protein_name_prefix+'0', curr_protein_name_prefix+'1',
                                         curr_protein_name_prefix+'2', label=2)
                    #
                    self.add_interaction(curr_concept_term, None, curr_protein_name_prefix+'1', curr_protein_name_prefix+'2',
                                         label=1)
                    self.add_interaction(curr_concept_term, None, curr_protein_name_prefix+'1', curr_protein_name_prefix+'2',
                                         label=0)
                    #
                    self.add_interaction(curr_concept_term, None, curr_protein_name_prefix+'2', curr_protein_name_prefix+'1',
                                         label=1)
                    self.add_interaction(curr_concept_term, None, curr_protein_name_prefix+'2', curr_protein_name_prefix+'1',
                                         label=0)
        if is_pickle:
            gefd.dump_pickle_data_joint_model(self.data, is_extend=False, is_synthetic=False)
        return self.data


if __name__ == '__main__':
    ca_obj = CanonicalAMRs()
    ca_obj.gen_amrs(is_pickle=True)

