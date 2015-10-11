from config_kernel import *
import time
import scipy.signal as ss
from constants import *
from config import *
import wordsegment as ws
import word2vec as wv
import constants_absolute_path as cap
import pickle
import config_hpcc as ch
from config_console_output import *
import config_processing as cp
import edge_labels as el
import edge_labels_propagation as elp


word2vec_model = None
cosine_similarity_map = {}
word_vectors_map = {}
#
is_amr_edge2vec = True
if is_amr_edge2vec:
    amr_edge2vec_map = None
#
is_dependencies2vec = True
if is_dependencies2vec:
    dependencies_edge2vec_map = None
#
if is_amr_edge2vec or is_dependencies2vec:
    edge2vec_map = {}
else:
    edge2vec_map = None
#
if cp.is_processing_amrs:
    print 'Loading word vectors into the python model ...'
    start_time = time.time()
    word2vec_model = wv.load(cap.absolute_path+'../../wordvectors/pubmed.bin')
    print 'The execution time for the loading was ', time.time()-start_time
    print 'word2vec_model.vocab', word2vec_model.vocab
    #
    # loading word vectors for edges
    if is_amr_edge2vec:
        print 'Loading amr edge vectors ...'
        amr_edge_label_word_vec_pickled_file_path = elp.get_edge_label_wordvector_file_path(is_amr=True)
        with open(cap.absolute_path+amr_edge_label_word_vec_pickled_file_path, 'r') as f_amr_edge_word2vec:
            amr_edge2vec_map = pickle.load(f_amr_edge_word2vec)
        assert amr_edge2vec_map is not None and amr_edge2vec_map
        edge2vec_map.update(amr_edge2vec_map)
    if is_dependencies2vec:
        print 'loading dependencies edge vectors ...'
        dependencies_edge_label_word_vec_pickled_file_path = elp.get_edge_label_wordvector_file_path(is_amr=False)
        with open(cap.absolute_path+dependencies_edge_label_word_vec_pickled_file_path, 'r') as f_dependencues_edge_word2vec:
            dependencies_edge2vec_map = pickle.load(f_dependencues_edge_word2vec)
        assert dependencies_edge2vec_map is not None and dependencies_edge2vec_map
        edge2vec_map.update(dependencies_edge2vec_map)


def get_edge_wordvector(edge_label):
    assert is_amr_edge2vec or is_dependencies2vec
    assert edge2vec_map is not None and edge2vec_map
    if edge_label is None:
        return None
    edge_label = edge_label.strip()
    edge_label_lower = edge_label.lower()
    edge_label_upper = edge_label.upper()
    #
    edge_label_inverse = el.get_inverse_of_edge_label(edge_label)
    edge_label_inverse_lower = edge_label_inverse.lower()
    edge_label_inverse_upper = edge_label_inverse.upper()
    try:
        if edge_label in edge2vec_map:
            return edge2vec_map[edge_label]
        elif edge_label_lower in edge2vec_map:
            return edge2vec_map[edge_label_lower]
        elif edge_label_upper in edge2vec_map:
            return edge2vec_map[edge_label_upper]
        elif edge_label_inverse in edge2vec_map:
            return edge2vec_map[edge_label_inverse]
        elif edge_label_inverse_lower in edge2vec_map:
            return edge2vec_map[edge_label_inverse_lower]
        elif edge_label_inverse_upper in edge2vec_map:
            return edge2vec_map[edge_label_inverse_upper]
        else:
            if debug:
                print '***********************************************'
                print 'error getting vector for edge label ', edge_label
                print edge_label
                print edge_label_lower
                print edge_label_upper
                print edge_label_inverse
                print edge_label_inverse_lower
                print edge_label_inverse_upper
                print '***********************************************'
    except UnicodeDecodeError as ude:
        if debug:
            print 'error getting vector for edge label ', edge_label
            print ude.message


def get_wordvector(word):
    if word is None:
        return None
    word = word.strip().strip('[').strip(']').strip('(').strip(')')
    word_lower = word.lower()
    word_upper = word.upper()
    try:
        if word_lower not in word_vectors_map:
            if debug:
                print 'getting word vector for ', word
            if word in word2vec_model.vocab:
                word_vectors_map[word_lower] = word2vec_model[word]
            #todo: if vocab us ensured to be lower case, this condition is not required
            elif word_lower in word2vec_model.vocab:
                word_vectors_map[word_lower] = word2vec_model[word_lower]
            elif word_upper in word2vec_model.vocab:
                word_vectors_map[word_lower] = word2vec_model[word_upper]
            else:
                if not concept_regexp.sub('', word):
                    return get_wordvector(alpha_regex.sub('', word))
                subwords = word.split()
                if len(subwords) == 1:
                    subwords = word.split(',')
                    if len(subwords) == 1:
                        subwords = word.split('/')
                        if len(subwords) == 1:
                            subwords = word.split(':')
                            if len(subwords) == 1:
                                subwords = word.split('-')
                                if len(subwords) == 1:
                                    subwords = word.split('_')
                                    if len(subwords) == 1:
                                        # print 'performing word segmentation on ', word
                                        subwords = ws.segment(word.encode('utf8'))
                                        if len(subwords) == 1:
                                            if not ch.is_hpcc:
                                                print 'could not get wordvector for ', word
                                            word_vectors_map[word_lower] = None
                if len(subwords) > 1:
                    curr_wordvec = None
                    for curr_subword in subwords:
                        curr_subword_vec = get_wordvector(curr_subword)
                        if curr_subword_vec is not None:
                            if curr_wordvec is None:
                                curr_wordvec = curr_subword_vec
                            else:
                                start_time = time.time()
                                curr_wordvec = ss.fftconvolve(curr_wordvec, curr_subword_vec, mode='same')
                                if debug:
                                    print 'performed fast fourier transform convolution on word vectors in {} seconds.'.format(time.time()-start_time)
                    word_vectors_map[word_lower] = curr_wordvec
        return word_vectors_map[word_lower]
    except UnicodeDecodeError as ude:
        if not ch.is_hpcc:
            print 'error getting word vector for ', word
        print ude.message
        word_vectors_map[word_lower] = None
        return word_vectors_map[word_lower]


