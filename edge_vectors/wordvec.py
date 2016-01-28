#
# Written by Sahil Garg (sahilgar@usc.edu, sahil@isi.edu, sahilvk87@gmail.com)
#
# Sahil Garg, Aram Galstyan, Ulf Hermjakob, and Daniel Marcu. Extracting biomolecular interactions using semantic parsing of biomedical text. In Proc. of AAAI, 2016.
#
# Copyright ISI-USC 2015
#
#
import time
import scipy.signal as ss
import wordsegment as ws
import word2vec as wv

from constants import *
from config import *
import constants_absolute_path as cap


word2vec_model = None
cosine_similarity_map = {}
word_vectors_map = {}
#
print 'Loading word vectors into the python model ...'
start_time = time.time()
word2vec_model = wv.load(cap.absolute_path+'./wordvectors/pubmed.bin')
print 'The execution time for the loading was ', time.time()-start_time
print 'word2vec_model.vocab', word2vec_model.vocab


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
        print 'error getting word vector for ', word
        print ude.message
        word_vectors_map[word_lower] = None
        return word_vectors_map[word_lower]

