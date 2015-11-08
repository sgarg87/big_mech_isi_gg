from config_console_output import *


#for classifying if a protein is a catalyst for an interaction
#
# Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is  0.15546875
# Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is  0.09100818
# Mean adjusted rand score for optimal lambda is  0.250386194997
# Max adjusted rand score for optimal lambda is  0.40026076535
# Min adjusted rand score for optimal lambda is  0.100161407565
# SD adjusted rand score for optimal lambda is  0.0455564149108
#
tuned_cs_fr_catalyst = 0.15546875
tuned_lambda_fr_catalyst = 0.09100818

#for classifying if a protein is domain (the one which goes state change)
#
# Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is  0.024833984375
# Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is  0.0900082
# Mean adjusted rand score for optimal lambda is  0.288497140535
# Max adjusted rand score for optimal lambda is  0.437830931775
# Min adjusted rand score for optimal lambda is  0.109084608661
# SD adjusted rand score for optimal lambda is  0.054906376017
#
# confusion matrix is better than case below
# [[106  39]
#  [ 29 126]]


tuned_cs_fr_domain = 0.0559375
tuned_lambda_fr_domain = 0.900091998


#multiclass case
# development performance (50% training and 50% test)
# inverse root only case
# Catalyst-Domain-None classification .....................................
# Optimal cosine threshold (tuned with cross validation using binary search on adjusted random score) is  0.0253125
# Optimal Lambda (tuned with cross validation using binary search on adjusted random score) is  0.9018
# Mean adjusted rand score for optimal lambda is  0.260619956662
# Max adjusted rand score for optimal lambda is  0.340504483095
# Min adjusted rand score for optimal lambda is  0.1882505488
# SD adjusted rand score for optimal lambda is  0.024607187543
#
# actual tuned value is commented below and instead a higher value is being used so as to decrease computational cost
# tuned_cs_fr_catalyst_domain = 0.0253125
tuned_cs_fr_catalyst_domain = 0.4
tuned_lambda_fr_catalyst_domain = 0.99

# joint case
# actual tuned value is commented below and instead a higher value is being used so as to decrease computational cost
# tuned_cs_fr_joint = 0.0253125
# Mean adjusted rand score for optimal lambda is  0.260619956662
# Max adjusted rand score for optimal lambda is  0.340504483095
# Min adjusted rand score for optimal lambda is  0.1882505488
# SD adjusted rand score for optimal lambda is  0.024607187543
#
# actual tuned value is commented below and instead a higher value is being used so as to decrease computational cost
# tuned_cs_fr_joint = 0.0253125
# tuned_cs_fr_joint = 0.4
tuned_cs_fr_joint = 0.6
tuned_lambda_fr_joint = 0.99
# test for 0.4, 0.9018
# Precision:  0.599513870958
# Recall:  0.626198083067
# F1:  0.593002538029
# zero_one_loss:  0.373801916933
# confusion matrix:  [[295  53   0]
#  [157  97   0]
#  [ 16   8   0]]
# roc auc :  0.573885043385
# adjusted random score:  0.0823847052182
#
# test with synthetic edges
# confusion matrix:  [[265  83   0]
#  [ 99 155   0]
#  [ 14  10   0]]
# Precision recall for positive label
#
# roc auc :  0.619209275534
# adjusted random score:  0.142474879955
#
# test with synthetic edges and model interactions
# confusion matrix:  [[334  14   0]
#  [196  58   0]
#  [ 24   0   0]]
# roc auc :  0.55985254276
# adjusted random score:  0.0570472736129

#
# protein state classifier
#
# random chance success should be about 0.26
# validation using cs=0.4, lam=0.99
#
# confusion matrix for mean ars  [[3473  445  226]
#  [ 394  733   56]
#  [  86   33  181]]
# adjusted rand score for mean ars  0.365663959873
# ****************************************************
# Minimum adjusted rand score for test  0.328371532686
# Mean adjusted rand score for test  0.367447686954
# SD adjusted rand score for test  0.0111527788781
# ****************************************************
#
# with concept as root node, here are results
# adjusted rand score for mean ars  0.303091293506
# confusion matrix for mean ars  [[3397  557  217]
#  [ 429  681   61]
#  [  91   41  152]]
# recall: 0.61
# precision: 0.55
# #
#
# after adding model data
#
# confusion matrix for mean ars  [[3762  268  164]
#  [ 536  922   63]
#  [  99   22  159]]
#
# Minimum adjusted rand score for test  0.38819435582
# Mean adjusted rand score for test  0.429168120469
# SD adjusted rand score for test  0.0123015538828#
#

tuned_cs_fr_protein_state = 0.6
tuned_lambda_fr_protein_state = 0.99

