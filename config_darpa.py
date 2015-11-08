from config_console_output import *


is_darpa = True

if is_darpa:
    all_proteins_of_interaction_in_model = False
    at_least_one_protein_of_interaction_in_model = True
    assert not (all_proteins_of_interaction_in_model and at_least_one_protein_of_interaction_in_model)

is_entity_in_model_by_identifier_only = False

is_connected_to_model_required = True

is_darpa_entity_type = True


