import config_train_extractor as cte
from config_console_output import *


file_name_svm_classifier_multiclass_not_joint = './file_name_svm_classifier_multiclass_not_joint'

if cte.is_model_interactions_graph_in_joint_train:
    file_name_svm_classifier_multiclass_joint = './file_name_svm_classifier_multiclass_joint_wd_model'
else:
    file_name_svm_classifier_multiclass_joint = './file_name_svm_classifier_multiclass_joint'
if cte.is_model_interactions_graph_in_protein_state_train:
    file_name_svm_classifier_protein_state = './file_name_svm_classifier_protein_state_wd_model'
else:
    file_name_svm_classifier_protein_state = './file_name_svm_classifier_protein_state'
