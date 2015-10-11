import model_json_interactions_to_amrs as mjia
import generate_canonical_amrs as gca
import gen_extractor_features_data as gtd
import biopax_model_obj as bmo
from config_console_output import *


def save():
    #
    ija_obj = mjia.InteractionsJSONToAMR()
    model_data = ija_obj.gen_amr_subgraphs_frm_model_interactions(bmo.bm_obj.json_objs)
    #
    ca_obj = gca.CanonicalAMRs()
    canonical_data = ca_obj.gen_amrs()
    #
    model_data[gtd.const_paths_map].update(canonical_data[gtd.const_paths_map])
    model_data[gtd.const_interaction_tuples_map].update(canonical_data[gtd.const_interaction_tuples_map])
    model_data[gtd.const_sentences_map].update(canonical_data[gtd.const_sentences_map])
    model_data[gtd.const_joint_labels_map].update(canonical_data[gtd.const_joint_labels_map])
    #
    gtd.dump_pickle_data_joint_model(model_data, is_extend=False, is_synthetic=False)
    #
    del ija_obj
    del ca_obj


if __name__ == '__main__':
    save()

