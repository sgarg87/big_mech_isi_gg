import graph_kernels as gk
import multiprocessing as mp
import numpy as np
import constants_absolute_path as cap
import time
from config_parallel_processing import *
import parallel_computing as pc
from config_console_output import *


def save_kernel_matrix(K, lam, cs):
    n1 = K.shape[0]
    n2 = K.shape[1]
    matrix_name = './gen_kernel_matrices/'
    matrix_name += 'graph_kernel_K_'
    matrix_name += str(n1)+'_'+str(n2)+'_lambda'+str(lam)
    if cs is not None:
        matrix_name += '_ct'+str(cs)
    start_time = time.time()
    np.save(cap.absolute_path+matrix_name, K)
    print 'Time to save the matrix was ', time.time()-start_time


def eval_kernel_parallel(amr_graphs1, amr_graphs2, lam, cosine_threshold):
    kernel_matrix_queue = [mp.Queue() for d in range(num_cores)]
    n1 = amr_graphs1.shape[0]
    n2 = amr_graphs2.shape[0]
    K = -1*np.ones(shape=(n1, n2))
    idx_range_parallel = pc.uniform_distribute_tasks_across_cores(n1, num_cores)
    processes = [
        mp.Process(
            target=eval_kernel_wrapper,
            args=(
                amr_graphs1[idx_range_parallel[currCore], :],
                amr_graphs2,
                lam,
                cosine_threshold,
                kernel_matrix_queue[currCore]
            )
        ) for currCore in range(num_cores)
    ]
    #start processes
    for process in processes:
        process.start()
    for currCore in range(num_cores):
        print('waiting for results from core ', currCore)
        # todo: this should not work, replace with mesh index
        result = kernel_matrix_queue[currCore].get()
        if isinstance(result, BaseException) or isinstance(result, OSError): #it means that subprocess has an error
            print 'a child processed has thrown an exception. raising the exception in the parent process to terminate the program'
            print 'one of the child processes failed, so killing all child processes'
            #kill all subprocesses
            for process in processes:
                if process.is_alive():
                    process.terminate() #assuming that the child process do not have its own children (those granchildren would be orphaned with terminate() if any)
            print 'killed all child processes'
            raise result
        else:
            K[idx_range_parallel[currCore], :] = result
        print('got results from core ', currCore)
    kernel_matrix_queue = None
    #wait for processes to complete
    for process in processes:
        process.join()
    assert np.all(K >= 0)
    save_kernel_matrix(K, lam, cosine_threshold)
    return K


def eval_kernel_wrapper(amr_graphs1, amr_graphs2, lam, cs, kernel_matrix_queue):
    try:
        K = gk.eval_graph_kernel_matrix(amr_graphs1, amr_graphs2, lam=lam, cosine_threshold=cs)
        assert K is not None
        assert K.shape[0] == amr_graphs1.shape[0]
        assert K.shape[1] == amr_graphs2.shape[0]
        #
        kernel_matrix_queue.put(K)
    except BaseException as e:
        print 'error in the subprocess (base exception)'
        print e
        kernel_matrix_queue.put(e)
    except OSError as ee:
        print 'error in the subprocess (os error)'
        print ee
        kernel_matrix_queue.put(ee)

