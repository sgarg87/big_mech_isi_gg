import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    bias_pos = -2.7
    bias_neg = -bias_pos
    #
    score_p = np.load('./score_test_pred_pos.npz')['arr_0']
    score_n = np.load('./score_test_pred_neg.npz')['arr_0']
    #
    t = np.arange(bias_pos, bias_neg, 0.001)
    x_pos = []
    x_neg = []
    for curr_t in t:
        x_pos.append(np.where(score_p > curr_t)[0].size)
        x_neg.append(np.where(score_n < curr_t)[0].size)
    #
    plt.plot(t, x_pos, 'b-')
    plt.plot(t, x_neg, 'r-')
    plt.xticks(np.arange(bias_pos, bias_neg, 0.1), fontsize=8)
    plt.gca().grid(True)
    plt.show()
