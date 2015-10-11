#!/bin/bash

#training

OMP_NUM_THREADS=1 python  ./gen_extractor_features_data.py 'True' "../../eval-auto-amr-1000papers_june2015/eval-auto-amr/eval-auto-amr_a_pmid_753_9035" 1 10 'True' 'True'

