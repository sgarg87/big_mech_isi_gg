from config_console_output import *


# lam_default = 0.5

is_root_kernel_default = True
# is_edge_label = False
kernel_computations_map_default = None
is_child_in_law = True
# is_diag_kernel_eval = True
#
# diag_only_fr_test = False

is_sparse = False
min_children_sparse = 3

is_reg_kernel = False
if is_reg_kernel:
    reg_lambda = 1e-5


#only one of these is true
is_svm = True
is_spectral_clustering = False
is_corex = False


is_nonlinear_kernel = False
# nonlinear_func = 'log'
nonlinear_func = 'pow'
nonlinear_pow = 0.5


num_trials_svm = 10000
training_frac = 0.5

is_tuning = True
lambda_tuning_branch = 10
lambda_tol = 1e-2
lambda_range_min = 0.99
lambda_range_max = 0.99
ct_tol = 1e-2
ct_range_min = 0.4
ct_range_max = 0.4


is_word_vectors = True
if is_word_vectors:
    is_word_vectors_round = False #similarity 1 is considered if cosine similarity above the threshold
    is_cosine_similarity_kernel_compact_supported = True
else:
    is_word_vectors_round = False #similarity 1 is considered if cosine similarity above the threshold
    is_cosine_similarity_kernel_compact_supported = False


is_pos_def_test = False


is_label_kernel = True
if is_label_kernel:
    t_words_pow_fr_label_kernel = 1

is_role_kernel = True
if is_role_kernel:
    t_words_pow_fr_role_kernel = 1


word_kernels = ['melkumyan', 'rbf_sparse']
word_kernel = word_kernels[1]

#todo: these parameters below should be tuned
rbf_bandwidth = 1

# if is_label_kernel:
#     #in this case, label and word correspond to two dimensions
#     sparse_kernel_v = 2
# else:
sparse_kernel_v = 1

is_laplace_kernel = False


both_name_type = False
if not both_name_type:
    is_name_else_type = False
else:
    is_name_else_type = None


is_svm_verbose = False


is_neighbor_kernel = False
if is_neighbor_kernel:
    is_neighbor_kernel_coeff_only = True
    is_parent_child_coeff = True
# else:
#     is_neighbor_kernel_coeff_only = None

is_save_matrix = True
is_save_matrix_img = False

is_normalize_each_kernel = True #otherwise matrix is normalized

is_inverse_centralize_amr = True


is_joint_amr_synthetic_edge = False
is_joint_amr_synthetic_role = True
if is_name_else_type is not None and not is_name_else_type:
    is_protein_state_subgraph_rooted_at_concept_node = False
else:
    is_protein_state_subgraph_rooted_at_concept_node = True
is_protein_state_amr_synthetic_edge = False
is_protein_state_amr_synthetic_role = True


is_word_vectors_edge = False

