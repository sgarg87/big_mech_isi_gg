import numpy as np
import file_paths_train_data as fptd
import pickle as p
import train_extractor as te


if __name__ == '__main__':
    amr_graphs, labels = te.get_data_joint(
        is_train=True,
        is_model_data=True,
        is_word_vectors=True,
        load_sentence_frm_dot_if_required=True,
        is_alternative_data=False,
        is_chicago_data=True
    )
    n = labels.size
    print 'labels.size', labels.size
    #
    print 'setting label 2 to value 0'
    idx_label2 = np.where(labels == 2)
    labels[idx_label2] = 0
    #
    print 'filtering out data containing only positive labels'
    positive_label_idx = np.where(labels == 1)[0]
    labels = labels[positive_label_idx]
    print 'labels.shape', labels.shape
    amr_graphs = amr_graphs[positive_label_idx, :]
    print 'amr_graphs.shape', amr_graphs.shape
    #
    file_path = fptd.processed_amr_graphs_lables_joint_train
    new_data = {}
    new_data['amr'] = amr_graphs
    new_data['label'] = labels
    #
    with open(file_path, 'wb') as f:
        print 'dumping ...'
        p.dump(new_data, f)
        print 'dumped'
