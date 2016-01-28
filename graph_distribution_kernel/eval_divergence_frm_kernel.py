#
# Written by Sahil Garg (sahilgar@usc.edu, sahil@isi.edu, sahilvk87@gmail.com)
#
# Sahil Garg, Aram Galstyan, Ulf Hermjakob, and Daniel Marcu. Extracting biomolecular interactions using semantic parsing of biomedical text. In Proc. of AAAI, 2016.
#
# Copyright ISI-USC 2015
#


import numpy as np
import time
from config import *


div_tol = 1e-2


def eval_divergence(Kii, Kjj, Kij, algo):
    #
    # kl divergence estimation using kernel density
    # maximum mean discrepancy method
    # cross kernels
    #
    assert Kii.shape[0] == Kii.shape[1] and len(Kii.shape) == 2
    assert Kjj.shape[0] == Kjj.shape[1] and len(Kjj.shape) == 2
    assert len(Kij.shape) == 2 and Kii.shape[0] == Kij.shape[0] and Kjj.shape[0] == Kij.shape[1]
    # maximum mean discrepancy
    if algo == 'mmd':
        return eval_max_mean_discrepancy(Kii, Kjj, Kij)
    if algo == 'k':
        return eval_distribution_kernel(Kij)
    # kl divergence with kernel density estimation
    elif algo == 'kl_kd':
        return eval_kl_div_kernel_density(Kii, Kjj, Kij)
    else:
        raise NotImplementedError


def eval_max_mean_discrepancy(Kii, Kjj, Kij):
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
        if -div_tol < maximum_mean_discrepancy < 0:
            maximum_mean_discrepancy = 0
        else:
            print 'Kii.mean()', Kii.mean()
            print 'Kjj.mean()', Kjj.mean()
            print 'Kij.mean()', Kij.mean()
            print 'maximum_mean_discrepancy', maximum_mean_discrepancy
            raise AssertionError
    if coarse_debug:
        print 'time to compute maximum mean discrepancy was {}'.format(time.time()-start_time)
    return maximum_mean_discrepancy


def eval_distribution_kernel(Kij):
    # Cross kernels
    # todo: rename the function as per the paper
    #
    return Kij.mean()


def eval_kl_div_kernel_density(Kii, Kjj, Kij):
    start_time = time.time()
    const_epsilon = 1e-30
    kl_ii_jj = np.log(np.divide(Kii.mean(1)+const_epsilon, Kij.mean(1)+const_epsilon)).mean()
    kl_jj_ii = np.log(np.divide(Kjj.mean(1)+const_epsilon, Kij.transpose().mean(1)+const_epsilon)).mean()
    kl = kl_ii_jj + kl_jj_ii
    if kl < 0:
        if -div_tol < kl < 0:
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
    if coarse_debug:
        print 'time to compute kl divergence with kernel density estimation was {}'.format(time.time()-start_time)
    return kl

