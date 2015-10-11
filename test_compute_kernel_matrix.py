import train_extractor as te
import graph_kernels as gk
import kernel_tuned_parameters as ktp
import scipy.sparse as ss
import save_sparse_scipy_matrices as sssm
import numpy as np
import config_kernel_matrices_format as ckmf


def get_test_data():
    amr_graphs, _ = te.get_data_joint(is_train=False, is_chicago_data=None, is_dependencies=False)
    return amr_graphs


def compute_kernel_matrix():
    amr_graphs = get_test_data()
    assert amr_graphs.shape[1] == 1
    #
    K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=ktp.tuned_lambda_fr_joint, cosine_threshold=ktp.tuned_cs_fr_joint)
    if ckmf.is_kernel_dtype_lower_precision:
        print K.dtype
        K = K.astype(ckmf.kernel_dtype_np, copy=False)
        print K.dtype
    #
    np.savez_compressed('./test_1', K)
    K = ss.csr_matrix(K)
    sssm.save_sparse_csr('./test1', K)
    #
    #
    K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=ktp.tuned_lambda_fr_joint, cosine_threshold=0.5)
    if ckmf.is_kernel_dtype_lower_precision:
        print K.dtype
        K = K.astype(ckmf.kernel_dtype_np, copy=False)
        print K.dtype
    np.savez_compressed('./test_2', K)
    K = ss.csr_matrix(K)
    sssm.save_sparse_csr('./test2', K)
    #
    #
    K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=ktp.tuned_lambda_fr_joint, cosine_threshold=0.6)
    if ckmf.is_kernel_dtype_lower_precision:
        print K.dtype
        K = K.astype(ckmf.kernel_dtype_np, copy=False)
        print K.dtype
    np.savez_compressed('./test_3', K)
    K = ss.csr_matrix(K)
    sssm.save_sparse_csr('./test3', K)
    #
    #
    K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=ktp.tuned_lambda_fr_joint, cosine_threshold=0.7)
    if ckmf.is_kernel_dtype_lower_precision:
        print K.dtype
        K = K.astype(ckmf.kernel_dtype_np, copy=False)
        print K.dtype
    np.savez_compressed('./test_4', K)
    K = ss.csr_matrix(K)
    sssm.save_sparse_csr('./test4', K)
    #
    #
    K = gk.eval_graph_kernel_matrix(amr_graphs, amr_graphs, lam=ktp.tuned_lambda_fr_joint, cosine_threshold=0.8)
    if ckmf.is_kernel_dtype_lower_precision:
        print K.dtype
        K = K.astype(ckmf.kernel_dtype_np, copy=False)
        print K.dtype
    np.savez_compressed('./test_5', K)
    K = ss.csr_matrix(K)
    sssm.save_sparse_csr('./test5', K)


if __name__ == '__main__':
    compute_kernel_matrix()
