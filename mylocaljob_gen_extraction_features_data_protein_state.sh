#!/bin/bash

#training

#'./bio-bmtr1/bio-bmtr1_bio.bmtr_0001' 1 21

#'./bio-bmtr1/bio-bmtr1_bio.bmtr_0002' 1 14

#'./bio-mskcc/bio-mskcc-57-40/bio-mskcc-57-40_bio.mskcc_0001' 1 26

#'./bio-mskcc/bio-mskcc-57-40/bio-mskcc-57-40_bio.mskcc_0001' 43 52

#'./bio-mskcc/bio-mskcc-57-40/bio-mskcc-57-40_bio.mskcc_0001' 57 57

#'./pmid-24651010/pmid-24651010_pmid_2465_1010' 1 263

#'./amr-release-bio/amr-release-bio_bio.mskcc_0001' 27 42

#'./amr-release-bio/amr-release-bio_bio.mskcc_0001' 53 56

#'./amr-release-bio/amr-release-bio_bio-kappa_0001' 1 27

#'./amr-release-bio/amr-release-bio_bio.bel_0002' 1 13

#'./amr-release-bio/amr-release-bio_bio.bmtr_0003' 1 21

#'./amr-release-bio/amr-release-bio_bio.ras_0001' 1 10

#'./amr-release-bio/amr-release-bio_bio.ras_0002' 1 5

#'./amr-release-bio/amr-release-bio_bio.ras_0003' 1 8

#this is same as the one encountered previously in ras or bel. so skipping it
#'./amr-release-bio/amr-release-bio_bio-exp_0001' 1 15


OMP_NUM_THREADS=1 python  ./gen_extractor_features_data.py 'True' './amr-release-bio/amr-release-bio_bio.bel_0002' 1 13 'True' 'True' 'True'

