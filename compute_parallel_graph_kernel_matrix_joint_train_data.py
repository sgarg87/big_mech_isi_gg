import train_extractor as te
import graph_kernels as gk
import kernel_tuned_parameters as ktp
import numpy as np
import constants_absolute_path as cap
import parallel_computing as pk
import scipy.sparse as ss
import save_sparse_scipy_matrices as sssm
import config_kernel_matrices_format as ckmf


is_alternative_data = False
is_save_sparse = True


def get_file_path(num_cores, curr_core):
    curr_core_kernel_file_path = './graph_kernel_matrix_joint_train_data_parallel/num_cores_{}_curr_core_{}'.format(num_cores, curr_core)
    return curr_core_kernel_file_path


def compute_kernel_matrix_joint_train_data(num_cores, curr_core):
    print 'No. of cores are ', num_cores
    print 'curr core is ', curr_core
    print 'cores start from index 0 ...'
    if is_alternative_data:
        amr_graphs, _ = te.get_data_joint(is_train=None, is_alternative_data=True)
    else:
        amr_graphs, _ = te.get_data_joint(is_train=True)
    n = amr_graphs.shape[0]
    assert amr_graphs.shape[1] == 1
    idx_range_parallel = pk.uniform_distribute_tasks_across_cores(n, num_cores)
    amr_graphs_curr_core = amr_graphs[idx_range_parallel[curr_core], :]
    K_curr_core = gk.eval_graph_kernel_matrix(amr_graphs_curr_core, amr_graphs, lam=ktp.tuned_lambda_fr_joint, cosine_threshold=ktp.tuned_cs_fr_joint)
    curr_core_kernel_file_path = get_file_path(num_cores, curr_core)
    #
    if ckmf.is_kernel_dtype_lower_precision:
        # print K_curr_core.dtype
        K_curr_core = K_curr_core.astype(ckmf.kernel_dtype_np, copy=False)
        # print K_curr_core.dtype
    #
    if not is_save_sparse:
        np.savez_compressed(cap.absolute_path+curr_core_kernel_file_path, K_curr_core)
    else:
        K_curr_core = ss.csr_matrix(K_curr_core)
        # print K_curr_core.dtype
        sssm.save_sparse_csr(cap.absolute_path+curr_core_kernel_file_path, K_curr_core)


def join_parallel_computed_kernel_matrices(num_cores):
    print 'No. of cores are ', num_cores
    print 'cores start from index 0 ...'
    #
    K_list = []
    n = None
    m = 0
    for curr_core in range(num_cores):
        curr_file_path = get_file_path(num_cores, curr_core)
        curr_core_K = np.load(cap.absolute_path+curr_file_path+'.npz')
        K_list.append(curr_core_K)
        if n is None:
            n = curr_core_K.shape[1]
        else:
            assert n == curr_core_K.shape[1]
        #
        m += curr_core_K.shape[0]
    assert n is not None
    assert m != 0
    assert m == n
    m = None
    #
    idx_range_parallel = pk.uniform_distribute_tasks_across_cores(n, num_cores)
    K = -1*np.ones(shape=(n, n))
    for curr_core in range(num_cores):
        K[np.meshgrid(idx_range_parallel[curr_core], range(n), indexing='ij')] = K_list[curr_core]
    K_list = None
    np.all(K >= 0)
    return K


if __name__ == '__main__':
    import sys
    num_cores = int(sys.argv[1])
    curr_core = int(sys.argv[2])
    compute_kernel_matrix_joint_train_data(num_cores, curr_core)
