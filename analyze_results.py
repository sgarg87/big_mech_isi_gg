import numpy as np


if __name__ == '__main__':
    labels_test_pred_prob = np.load('./labels_test_pred_prob.npz')['arr_0']
    #
    print 'labels_test_pred_prob.sum()', labels_test_pred_prob.sum()
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.1)] = 1
    print labels_test_pred.sum()
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.2)] = 1
    print labels_test_pred.sum()
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.3)] = 1
    print labels_test_pred.sum()
    #
    print '***************************************************'
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.4)] = 1
    print labels_test_pred.sum()
    #
    print '***************************************************'
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.5)] = 1
    print labels_test_pred.sum()
    #
    print '***************************************************'
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.6)] = 1
    print labels_test_pred.sum()
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.7)] = 1
    print labels_test_pred.sum()
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.8)] = 1
    print labels_test_pred.sum()
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.9)] = 1
    print labels_test_pred.sum()
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.95)] = 1
    print labels_test_pred.sum()
    #
    labels_test_pred = np.zeros(labels_test_pred_prob.shape)
    labels_test_pred[np.where(labels_test_pred_prob > 0.99)] = 1
    print labels_test_pred.sum()
