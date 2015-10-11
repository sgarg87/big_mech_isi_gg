#!/bin/bash


#optimal lambda
OMP_NUM_THREADS=1 python  ./graph_kernels.py '' -1 -1 '' 0.000990000099802 0.502822265625

#this one is appropriate when we use sparse covariance function for word similarity
#OMP_NUM_THREADS=1 python  ./graph_kernels.py '' -1 -1 '' 1e-05 0.600678585205

#OMP_NUM_THREADS=1 python  ./graph_kernels.py '' -1 -1 '' 0.0009999802





