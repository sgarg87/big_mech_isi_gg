#!/bin/bash

(
    python ./compute_parallel_graph_kernel_matrix_joint_train_data.py 1 0 3
)&

wait

