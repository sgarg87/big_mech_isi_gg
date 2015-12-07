import train_extractor as te
import graph_kernels as gk
import kernel_tuned_parameters as ktp
import numpy as np
import constants_absolute_path as cap
import parallel_computing as pk
import scipy.sparse as ss
import save_sparse_scipy_matrices as sssm
import config_kernel_matrices_format as ckmf
import time
import parallel_kernel_eval as pke


is_save_sparse = True


def get_file_path(num_cores, curr_core=None):
    curr_core_kernel_file_path = './graph_kernel_matrix_joint_train_data_parallel/'
    if curr_core is not None:
        curr_core_kernel_file_path += 'num_cores_{}_curr_core_{}'
        curr_core_kernel_file_path = curr_core_kernel_file_path.format(num_cores, curr_core)
    else:
        curr_core_kernel_file_path += 'num_cores_{}'
        curr_core_kernel_file_path = curr_core_kernel_file_path.format(num_cores)
    return curr_core_kernel_file_path


# def get_train_data():
#     amr_graphs, _ = te.get_processed_train_joint_data()
#     return amr_graphs


def get_train_data():
    amr_graphs, _ = te.get_data_joint(is_train=True, load_sentence_frm_dot_if_required=False)
    return amr_graphs


def compute_kernel_matrix_joint_train_data(num_cores, curr_core, num_threads, amr_graphs, amr_graphs2, is_core_sel):
    print 'No. of cores are ', num_cores
    print 'curr core is ', curr_core
    print 'cores start from index 0 ...'
    #
    if amr_graphs2 is not None:
        print 'amr_graphs2.shape', amr_graphs2.shape
    #
    if amr_graphs2 is None:
        amr_graphs2 = amr_graphs
    #
    n = amr_graphs.shape[0]
    assert amr_graphs.shape[1] == 1
    #
    if is_core_sel:
        idx_range_parallel = pk.uniform_distribute_tasks_across_cores(n, num_cores)
        amr_graphs = amr_graphs[idx_range_parallel[curr_core], :]
    #
    if num_threads == 1:
        K_curr_core = gk.eval_graph_kernel_matrix(
            amr_graphs,
            amr_graphs2,
            lam=ktp.tuned_lambda_fr_joint,
            cosine_threshold=ktp.tuned_cs_fr_joint,
            is_sparse=True,
            is_normalize=True
        )
    else:
        K_curr_core = pke.eval_kernel_parallel(
            amr_graphs,
            amr_graphs2,
            lam=ktp.tuned_lambda_fr_joint,
            cosine_threshold=ktp.tuned_cs_fr_joint,
            num_cores=num_threads)
    #
    curr_core_kernel_file_path = get_file_path(num_cores, curr_core)
    #
    # if ckmf.is_kernel_dtype_lower_precision:
    #     K_curr_core = K_curr_core.astype(ckmf.kernel_dtype_np)
    #
    if not is_save_sparse:
        np.savez_compressed(cap.absolute_path+curr_core_kernel_file_path, K_curr_core)
    else:
        K_curr_core = K_curr_core.tocsr()
        # print K_curr_core.dtype
        sssm.save_sparse_csr(cap.absolute_path+curr_core_kernel_file_path, K_curr_core)


def join_parallel_computed_kernel_matrices(num_cores):
    print 'No. of cores are ', num_cores
    print 'cores start from index 0 ...'
    #
    K_list = []
    n = None
    m = 0
    #
    k_dtype = None
    for curr_core in range(num_cores):
        curr_file_path = get_file_path(num_cores, curr_core)
        print 'curr_file_path', curr_file_path
        curr_core_K = np.load(cap.absolute_path+curr_file_path+'.npy')
        #
        if k_dtype is None:
            k_dtype = curr_core_K.dtype
        else:
            assert k_dtype == curr_core_K.dtype
        print curr_core_K.shape
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
    K = -1*np.ones(shape=(n, n), dtype=k_dtype)
    for curr_core in range(num_cores):
        K[np.meshgrid(idx_range_parallel[curr_core], range(n), indexing='ij')] = K_list[curr_core]
    #
    K_list = None
    np.all(K >= 0)
    return K


def join_parallel_computed_kernel_matrices_sparse(num_cores):
    print 'No. of cores are ', num_cores
    print 'cores start from index 0 ...'
    #
    n = None
    m = 0
    #
    K_map = {}
    #
    for curr_core in range(num_cores):
        curr_file_path = get_file_path(num_cores, curr_core)
        print 'curr_file_path', curr_file_path
        #
        print 'loading ...'
        start_time = time.time()
        curr_core_K = sssm.load_sparse_csr(cap.absolute_path+curr_file_path+'.npz')
        print time.time()-start_time
        print 'loaded'
        #
        K_map[curr_core] = curr_core_K
        #
        print 'curr_core_K.shape', curr_core_K.shape
        print 'curr_core_K.nnz', curr_core_K.nnz
        #
        m += curr_core_K.shape[0]
        #
        if n is None:
            n = curr_core_K.shape[1]
        else:
            assert n == curr_core_K.shape[1]
    #
    print 'n', n
    assert n is not None
    print 'm', m
    assert m is not None
    #
    idx_range_parallel = pk.uniform_distribute_tasks_across_cores(m, num_cores)
    data = []
    row = []
    col = []
    #
    for curr_core in range(num_cores):
        curr_core_K = K_map[curr_core]
        K_map.pop(curr_core, None)
        #
        curr_row = np.array(idx_range_parallel[curr_core])
        n_r = curr_row.size
        curr_col = np.arange(n)
        n_c = curr_col.size
        #
        curr_data = curr_core_K.toarray().flatten().astype(np.float16)
        curr_core_K = None
        #
        print 'curr_data', curr_data
        curr_non_sparse_idx = np.where(curr_data > 1e-2)
        print 'curr_non_sparse_idx', curr_non_sparse_idx
        print 'curr_non_sparse_idx[0].dtype', curr_non_sparse_idx[0].dtype
        print 'len(curr_non_sparse_idx)', len(curr_non_sparse_idx)
        curr_data = curr_data[curr_non_sparse_idx]
        data.append(curr_data)
        curr_data = None
        #
        curr_row = np.tile(curr_row.reshape(n_r, 1), n_c).flatten()
        curr_row = curr_row[curr_non_sparse_idx]
        row.append(curr_row)
        curr_row = None
        #
        curr_col = np.tile(curr_col, n_r)
        curr_col = curr_col[curr_non_sparse_idx]
        col.append(curr_col)
        curr_col = None
        #
        curr_non_sparse_idx = None
    print 'merging all sparse data ...'
    #
    start_time = time.time()
    data = np.hstack(tuple(data))
    print time.time()-start_time
    #
    start_time = time.time()
    row = np.hstack(tuple(row))
    print time.time()-start_time
    #
    start_time = time.time()
    col = np.hstack(tuple(col))
    print time.time()-start_time
    #
    K = ss.coo_matrix((data, (row, col)), shape=(m, n), dtype=np.float16)
    data = None
    row = None
    col = None
    K = K.tocsr()
    #
    print K.nnz
    print K.data
    #
    sssm.save_sparse_csr(get_file_path(num_cores), K)
    #
    return K


def join_parallel_computed_kernel_matrices_sparse_wd_dok(num_cores):
    raise DeprecationWarning
    print 'No. of cores are ', num_cores
    print 'cores start from index 0 ...'
    #
    n = None
    m = 0
    k_dtype = None
    #
    K_map = {}
    #
    for curr_core in range(num_cores):
        curr_file_path = get_file_path(num_cores, curr_core)
        print 'curr_file_path', curr_file_path
        #
        print 'loading ...'
        start_time = time.time()
        curr_core_K = sssm.load_sparse_csr(cap.absolute_path+curr_file_path+'.npz')
        print time.time()-start_time
        print 'loaded'
        #
        curr_core_K = curr_core_K.todok()
        #
        K_map[curr_core] = curr_core_K
        #
        if k_dtype is None:
            assert n is None
            #
            k_dtype = curr_core_K.dtype
            n = curr_core_K.shape[1]
            assert n != 0
        else:
            assert k_dtype == curr_core_K.dtype
            assert n is not None
            assert n == curr_core_K.shape[1]
        #
        print curr_core_K.shape
        #
        m += curr_core_K.shape[0]
    #
    assert m != 0
    #
    idx_range_parallel = pk.uniform_distribute_tasks_across_cores(m, num_cores)
    #
    K = ss.dok_matrix((m, n), dtype=k_dtype)
    #
    for curr_core in range(num_cores):
        K[idx_range_parallel[curr_core], :] = K_map[curr_core]
        K_map[curr_core] = None
    #
    K = K.tocsr()
    #
    print K.nnz
    sssm.save_sparse_csr(get_file_path(num_cores), K)
    #
    return K


def get_chicago_data_fr_core(num_cores, curr_core):
    print 'num_cores', num_cores
    print 'curr_core', curr_core
    #
    file_path = './amr_graphs_chicago'
    file_path += '__{}_{}'.format(num_cores, curr_core)
    #
    start_time = time.time()
    print 'loading curr core chicago amrs ...'
    amr_graphs_chicago = np.load(cap.absolute_path+file_path+'.npy')
    print 'loaded in ', time.time()-start_time
    #
    return amr_graphs_chicago


def get_chicago_data(is_filter=False):
    if is_filter:
        start_time = time.time()
        print 'loading filtered chicago amrs ...'
        amr_graphs_chicago_filtered = np.load(cap.absolute_path+'./amr_graphs_chicago_filtered.npy')
        print 'loaded in ', time.time()-start_time
        return amr_graphs_chicago_filtered
    else:
        start_time = time.time()
        print 'loading all chicago amrs ...'
        amr_graphs_chicago = np.load(cap.absolute_path+'./amr_graphs_chicago.npy')
        print 'loaded in ', time.time()-start_time
        return amr_graphs_chicago


if __name__ == '__main__':
    import sys
    num_cores = int(sys.argv[1])
    curr_core = int(sys.argv[2])
    #
    if len(sys.argv) > 3:
        num_threads = int(sys.argv[3])
    else:
        num_threads = 1
    #
    is_train_matrix_compute = True
    load_amr_fr_core_first = False
    #
    if is_train_matrix_compute:
        if load_amr_fr_core_first:
            raise NotImplementedError
        amr_graphs = get_train_data()
        amr_graphs2 = None
    else:
        if not load_amr_fr_core_first:
            amr_graphs = get_chicago_data()
        else:
            amr_graphs = get_chicago_data_fr_core(num_cores, curr_core)
        assert amr_graphs is not None and amr_graphs.size > 0
        #
        amr_graphs2 = get_train_data()
        assert amr_graphs2 is not None and amr_graphs2.size > 0
    #
    compute_kernel_matrix_joint_train_data(num_cores, curr_core, num_threads, amr_graphs, amr_graphs2,
                                           is_core_sel=(not load_amr_fr_core_first))
