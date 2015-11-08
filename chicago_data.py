import semi_automated_extraction_features_data as saefd
import train_extractor as te
import numpy as np
import constants_absolute_path as cap
import time
import parallel_computing as pk


def get_chicago_filtered_ids_fr_test():
    filtered_sentence_ids = ['ID'+str(i+1) for i in range(4000)]
    return filtered_sentence_ids


def save_chicago_data_fr_cores(num_cores):
    file_path = './amr_graphs_chicago'
    #
    start_time = time.time()
    print 'loading ...'
    amr_graphs_chicago = np.load(cap.absolute_path+file_path+'.npy')
    print 'loaded in ', time.time()-start_time
    #
    n = amr_graphs_chicago.shape[0]
    assert amr_graphs_chicago.shape[1] == 1
    #
    idx_range_parallel = pk.uniform_distribute_tasks_across_cores(n, num_cores)
    #
    for curr_core in range(num_cores):
        curr_path = file_path+'__{}_{}'.format(num_cores, curr_core)
        print curr_path
        #
        amr_graphs_curr_core = amr_graphs_chicago[idx_range_parallel[curr_core], :]
        print amr_graphs_curr_core.shape
        #
        start_time = time.time()
        print 'saving ...'
        np.save(cap.absolute_path+curr_path, amr_graphs_curr_core)
        print 'saved in ', time.time()-start_time


def get_chicago_data(is_filter=False):
    chicago_data = saefd.load_chicago_data_joint()
    amr_graphs_chicago = te.process_pickled_data(chicago_data, is_labels=False)
    chicago_data = None
    assert amr_graphs_chicago is not None
    assert amr_graphs_chicago.shape[0] != 0
    assert amr_graphs_chicago.size != 0
    print 'amr_graphs_chicago.shape', amr_graphs_chicago.shape
    #
    np.save(cap.absolute_path+'./amr_graphs_chicago', amr_graphs_chicago)
    #
    if is_filter:
        # filtering data based on sentence ids
        # for chicago data, sentence ids range from 1 to 40,000 approx.
        # here for a test purpose, we are taking first 4000 sentences
        #
        filtered_sentence_ids = get_chicago_filtered_ids_fr_test()
        print 'filtered_sentence_ids', filtered_sentence_ids
        #
        m = amr_graphs_chicago.shape[0]
        print 'm', m
        #
        import semi_automated_extraction_features_chicago_data as saefcd
        filtered_graphs_idx = []
        for curr_idx in range(m):
            curr_amr = amr_graphs_chicago[curr_idx, 0]
            curr_path = curr_amr['path']
            print 'curr_path', curr_path
            curr_sentence_id = saefcd.get_passage_sentence_id_frm_joint_graph_path(curr_path)
            curr_sentence_id = 'ID'+str(curr_sentence_id)
            print 'curr_sentence_id', curr_sentence_id
            if curr_sentence_id in filtered_sentence_ids:
                filtered_graphs_idx.append(curr_idx)
        #
        print 'filtered_graphs_idx', filtered_graphs_idx
        #
        amr_graphs_chicago_filtered = amr_graphs_chicago[filtered_graphs_idx, :]
        amr_graphs_chicago = None
        print 'amr_graphs_chicago_filtered.shape', amr_graphs_chicago_filtered.shape
        #
        np.save('./amr_graphs_chicago_filtered', amr_graphs_chicago_filtered)
        #
        return amr_graphs_chicago_filtered
    else:
        return amr_graphs_chicago


if __name__ == '__main__':
    # get_chicago_data(is_filter=True)
    save_chicago_data_fr_cores(240)
