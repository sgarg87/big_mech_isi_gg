#
# Written by Sahil Garg (sahilgar@usc.edu, sahil@isi.edu, sahilvk87@gmail.com)
#
# Sahil Garg, Aram Galstyan, Ulf Hermjakob, and Daniel Marcu. Extracting biomolecular interactions using semantic parsing of biomedical text. In Proc. of AAAI, 2016.
#
# Copyright ISI-USC 2015
#


import time
import numpy as np
from .. import config


class GraphDistributionDivergence:
    def __init__(self):
        self.div_tol = 1e-2

    def evaluate_maximum_mean_discrepancy(self, Kii, Kjj, Kij):
        #
        # evaluate maximum mean discrepancy between two graph distribution
        #
        # Kii is graph kernel matrix on graph samples in distribution i
        # Kjj is graph kernel matrix on graph samples in distribution j
        # Kij is a matrix of graph kernels between samples of distribution i and samples of distribution j
        #
        assert Kii.shape[0] == Kii.shape[1] and len(Kii.shape) == 2
        assert Kjj.shape[0] == Kjj.shape[1] and len(Kjj.shape) == 2
        assert len(Kij.shape) == 2 and Kii.shape[0] == Kij.shape[0] and Kjj.shape[0] == Kij.shape[1]
        #
        start_time = time.time()
        print Kii
        print Kjj
        print Kij
        maximum_mean_discrepancy = Kii.mean() + Kjj.mean() - 2*Kij.mean()
        #
        print 'Kii.mean()', Kii.mean()
        print 'Kjj.mean()', Kjj.mean()
        print 'Kij.mean()', Kij.mean()
        #
        if maximum_mean_discrepancy < 0:
            if -self.div_tol < maximum_mean_discrepancy < 0:
                maximum_mean_discrepancy = 0
            else:
                print 'Kii.mean()', Kii.mean()
                print 'Kjj.mean()', Kjj.mean()
                print 'Kij.mean()', Kij.mean()
                print 'maximum_mean_discrepancy', maximum_mean_discrepancy
                raise AssertionError
        if config.coarse_debug:
            print 'time to compute maximum mean discrepancy was {}'.format(time.time()-start_time)
        return maximum_mean_discrepancy

    def evaluate_cross_kernels(self, Kij):
        #
        # evaluate Cross kernels divergence between two graph distributions
        #
        # Kij is a matrix of graph kernels between samples of distribution i and samples of distribution j
        #
        assert len(Kij.shape) == 2
        #
        return Kij.mean()

    def evaluate_kl_divergence_wd_graph_kernel_density(self, Kii, Kjj, Kij):
        #
        # evaluate KL divergence between two graph distributions using graph kernel density
        #
        # Kii is graph kernel matrix on graph samples in distribution i
        # Kjj is graph kernel matrix on graph samples in distribution j
        # Kij is a matrix of graph kernels between samples of distribution i and samples of distribution j
        #
        assert Kii.shape[0] == Kii.shape[1] and len(Kii.shape) == 2
        assert Kjj.shape[0] == Kjj.shape[1] and len(Kjj.shape) == 2
        assert len(Kij.shape) == 2 and Kii.shape[0] == Kij.shape[0] and Kjj.shape[0] == Kij.shape[1]
        #
        start_time = time.time()
        const_epsilon = 1e-30
        kl_ii_jj = np.log(np.divide(Kii.mean(1)+const_epsilon, Kij.mean(1)+const_epsilon)).mean()
        kl_jj_ii = np.log(np.divide(Kjj.mean(1)+const_epsilon, Kij.transpose().mean(1)+const_epsilon)).mean()
        kl = kl_ii_jj + kl_jj_ii
        if kl < 0:
            if -self.div_tol < kl < 0:
                kl = 0
            else:
                print 'Kii.mean(1)+const_epsilon', Kii.mean(1)+const_epsilon
                print 'Kjj.mean(1)+const_epsilon', Kjj.mean(1)+const_epsilon
                print 'Kij.mean(1)+const_epsilon', Kij.mean(1)+const_epsilon
                print 'Kij.transpose().mean(1)+const_epsilon', Kij.transpose().mean(1)+const_epsilon
                print 'kl_ii_jj', kl_ii_jj
                print 'kl_jj_ii', kl_jj_ii
                print 'kl', kl
                raise AssertionError
        if config.coarse_debug:
            print 'time to compute kl divergence with kernel density estimation was {}'.format(time.time()-start_time)
        return kl

