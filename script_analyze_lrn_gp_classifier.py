import constants_trained_svm_kernel_file_paths as ctskfp
import time
import pickle
import numpy as np
import subprocess as sp
import extract_from_amr_dot as ead
import postprocess_amr_text as pat
import random as r
import matplotlib.pyplot as plt


def input():
    curr_j = raw_input('Enter any number')
    return curr_j


if __name__ == '__main__':
        file_name = ctskfp.file_name_svm_classifier_multiclass_joint + '.pickle'
        print 'loading the model'
        start_time = time.time()
        with open(file_name, 'r') as h:
            trained_clf = pickle.load(h)
        print 'Time to load the trained svm model (and training samples) was ', time.time()-start_time
        svm_clf = trained_clf['model']
        assert svm_clf == 'gp'
        gp_weights = trained_clf['gp_weights']
        print 'gp_weights', gp_weights.shape
        kernel_normalization = trained_clf['kernel_normalization']
        #
        plt.plot(kernel_normalization, gp_weights, 'rx')
        plt.savefig('./temp_analyze_kernel_normalization_gp_weights.pdf', format='pdf', dpi=300)
        plt.close()
        plt.plot(gp_weights, 'rx')
        plt.savefig('./temp_analyze_kernel_gp_weights.pdf', format='pdf', dpi=300)
        plt.close()
        #
        train_amr_graphs = trained_clf['train_amrs']
        #
        weight_idx_list = np.where(np.absolute(gp_weights) < 0.001)[0]
        print 'weight_idx_list.shape', weight_idx_list.shape
        for curr_idx in weight_idx_list:
            print '*******************'
            print 'curr_idx', curr_idx
            print gp_weights[curr_idx]
            print kernel_normalization[curr_idx]
            #
            curr_amr_map = train_amr_graphs[curr_idx, 0]
            #
            curr_path = curr_amr_map['path']
            print 'curr_path', curr_path
            #
            curr_nodes_list = curr_amr_map['nodes']
            #
            curr_sentence = curr_amr_map['text']
            curr_sentence = pat.post_process_amr_text_sentence(curr_sentence)
            #
            dot_graph = ead.nodes_to_dot(curr_nodes_list, dot_file_path=None, sentence=curr_sentence)
            #
            dot_pdf_file_path = './temp_analyze'+str(r.randint(1, 1000000))+'.pdf'
            dot_graph.write_pdf(dot_pdf_file_path)
            #
            pdf_prg_id = sp.call(['open', dot_pdf_file_path])
            #
            input()
            #
            # os.kill(pdf_prg_id, 0)

