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
from .. import constants
from .. import config
from .. import constants_absolute_path as cap


class WordVectors:
    def __init__(self):
        self.word2vec_model = None
        self.cosine_similarity_map = {}
        self.word_vectors_map = {}
        #
        print 'Loading word vectors into the python model ...'
        start_time = time.time()
        self.word2vec_model = wv.load(cap.absolute_path+'./wordvectors/pubmed.bin')
        print 'The execution time for the loading was ', time.time()-start_time
        print 'word2vec_model.vocab', self.word2vec_model.vocab

    def get_word_vector(self, word):
        if word is None:
            return None
        word = word.strip().strip('[').strip(']').strip('(').strip(')')
        word_lower = word.lower()
        word_upper = word.upper()
        try:
            if word_lower not in self.word_vectors_map:
                if config.debug:
                    print 'getting word vector for ', word
                if word in self.word2vec_model.vocab:
                    self.word_vectors_map[word_lower] = self.word2vec_model[word]
                #todo: if vocab us ensured to be lower case, this condition is not required
                elif word_lower in self.word2vec_model.vocab:
                    self.word_vectors_map[word_lower] = self.word2vec_model[word_lower]
                elif word_upper in self.word2vec_model.vocab:
                    self.word_vectors_map[word_lower] = self.word2vec_model[word_upper]
                else:
                    if not constants.concept_regexp.sub('', word):
                        return self.get_word_vector(constants.alpha_regex.sub('', word))
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
                                                self.word_vectors_map[word_lower] = None
                    if len(subwords) > 1:
                        curr_wordvec = None
                        for curr_subword in subwords:
                            curr_subword_vec = self.get_word_vector(curr_subword)
                            if curr_subword_vec is not None:
                                if curr_wordvec is None:
                                    curr_wordvec = curr_subword_vec
                                else:
                                    start_time = time.time()
                                    curr_wordvec = ss.fftconvolve(curr_wordvec, curr_subword_vec, mode='same')
                                    if config.debug:
                                        print 'performed fast fourier transform convolution on word vectors in {} seconds.'.format(time.time()-start_time)
                        self.word_vectors_map[word_lower] = curr_wordvec
            return self.word_vectors_map[word_lower]
        except UnicodeDecodeError as ude:
            print 'error getting word vector for ', word
            print ude.message
            self.word_vectors_map[word_lower] = None
            return self.word_vectors_map[word_lower]

