import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    org_max_lkl_amr_only = [
               0.22499999999999998,
               0.10526315789473684,
               0.0,
               0.39999999999999997,
               0.5,
               0.20408163265306123,
               0.2654867256637168,
               0.28571428571428575,
               0.14285714285714285,
               0.6666666666666666,
               0.8571428571428571,
               0.39999999999999997,
               0.24615384615384617,
               0.33333333333333337,
               0.28571428571428575,
               0.33333333333333337,
               0.25,
               0.3714285714285715,
               0.2772277227722772,
               0.3333333333333333,
               0.4285714285714285,
               0.3714285714285714,
               0.13333333333333333,
               0.2978723404255319,
               0.2857142857142857,
               1.0,
               0.4,
               0.1111111111111111,
               0.0,
               0.3037974683544304,
               1.0,
               0.23076923076923078,
               0.4,
               0.3076923076923077,
               0.16666666666666666,
               0.05555555555555555,
               0.2608695652173913,
               0.375,
               0.37037037037037035,
               0.2857142857142857,
               0.3181818181818182,
               0.5454545454545454,
               0.4444444444444445,
               0.4,
               0.0,
               0.4085106382978724,
               0.0
          ]

    org_max_lkl_amr_only = np.array(org_max_lkl_amr_only)

    org_max_lkl_sdg_only = [
          ]
    org_max_lkl_sdg_only = np.array(org_max_lkl_sdg_only)


    org_max_lkl = [
              ]
    org_max_lkl = np.array(org_max_lkl)

    org_max_lkl_amr_sdg_zero = [
          ]
    org_max_lkl_amr_sdg_zero = np.array(org_max_lkl_amr_sdg_zero)

    mmd = [
              ]
    mmd = np.array(mmd)

    mmd_amr_sdg_zero = [
          ]
    mmd_amr_sdg_zero = np.array(mmd_amr_sdg_zero)

    mmd_amr = [
               0.20253164556962025,
               0.08333333333333333,
               0.0,
               0.3636363636363636,
               0.8,
               0.09090909090909091,
               0.2456140350877193,
               0.28571428571428575,
               0.13333333333333333,
               0.6666666666666666,
               0.7499999999999999,
               0.4615384615384615,
               0.4528301886792453,
               0.33333333333333337,
               0.0,
               0.5714285714285715,
               0.2,
               0.3636363636363637,
               0.2037037037037037,
               0.2857142857142857,
               0.3287671232876712,
               0.2857142857142857,
               0.2,
               0.2978723404255319,
               0.19999999999999998,
               1.0,
               0.3928571428571429,
               0.08,
               0.0,
               0.23214285714285712,
               0.0,
               0.3703703703703704,
               0.34545454545454546,
               0.32432432432432434,
               0.16666666666666666,
               0.08849557522123895,
               0.23076923076923075,
               0.3703703703703703,
               0.41379310344827586,
               0.33333333333333326,
               0.32653061224489793,
               0.42857142857142855,
               0.3636363636363636,
               0.4,
               0.0,
               0.3744292237442922,
               0.0
          ]
    mmd_amr = np.array(mmd_amr)
    #
    mmd_sdg = [
          ]
    mmd_sdg = np.array(mmd_sdg)
    #

    kl_kd_amr = [
               0.11594202898550725,
               0.1111111111111111,
               0.0,
               0.2,
               0.6666666666666666,
               0.12121212121212122,
               0.12631578947368421,
               0.4,
               0.16666666666666666,
               0.6666666666666666,
               0.5714285714285715,
               0.4615384615384615,
               0.15384615384615385,
               0.33333333333333337,
               0.0,
               0.33333333333333337,
               0.2857142857142857,
               0.3137254901960784,
               0.11904761904761904,
               0.3333333333333333,
               0.23076923076923078,
               0.12000000000000001,
               0.13333333333333333,
               0.17142857142857143,
               0.25,
               1.0,
               0.3703703703703704,
               0.14285714285714288,
               0.0,
               0.12500000000000003,
               0.0,
               0.1818181818181818,
               0.2857142857142857,
               0.0,
               0.08,
               0.0,
               0.26666666666666666,
               0.19047619047619047,
               0.47058823529411764,
               0.22222222222222224,
               0.25,
               0.22222222222222224,
               0.5714285714285715,
               0.3333333333333333,
               0.0,
               0.2641509433962264,
               0.0
          ]
    kl_kd_amr = np.array(kl_kd_amr)

    positive_label_perc = [
               0.12845849802371542,
               0.1650485436893204,
               0.040816326530612242,
               0.079207920792079209,
               0.2857142857142857,
               0.12863070539419086,
               0.037313432835820892,
               0.17647058823529413,
               0.083333333333333329,
               0.025000000000000001,
               0.12,
               0.14285714285714285,
               0.10749185667752444,
               0.20000000000000001,
               0.043478260869565216,
               0.16129032258064516,
               0.028571428571428571,
               0.099378881987577633,
               0.19451371571072318,
               0.06569343065693431,
               0.24064171122994651,
               0.21126760563380281,
               0.065217391304347824,
               0.11372549019607843,
               0.015873015873015872,
               0.0053763440860215058,
               0.1005586592178771,
               0.080882352941176475,
               0.0,
               0.073008849557522126,
               0.0,
               0.13533834586466165,
               0.072639225181598058,
               0.082802547770700632,
               0.083720930232558138,
               0.025718608169440244,
               0.012448132780082987,
               0.064777327935222673,
               0.060606060606060608,
               0.097222222222222224,
               0.14691943127962084,
               0.060344827586206899,
               0.020920502092050208,
               0.12844036697247707,
               0.0,
               0.029826014913007456,
               0.054054054054054057
          ]

    positive_label_perc = np.array(positive_label_perc)

    #
    train_test_div = [
          0.032822149915599654,
          0.028364678210863855,
          0.16843592407800165,
          0.028723968485635555,
          0.091012174014000288,
          0.020482491576307917,
          0.011994103956618138,
          0.10480389566668211,
          0.044577664853705415,
          0.21122275929793802,
          0.072583878224656603,
          0.19726257142225362,
          0.027078953704080078,
          0.10040443414736537,
          0.05008325287179826,
          0.1154737389593585,
          0.022267854232902895,
          0.014930369807308533,
          0.019193520483342608,
          0.028197062653570144,
          0.015368024645410065,
          0.02611815700034395,
          0.012476973461723374,
          0.028325661264337589,
          0.052513593541453074,
          0.04097240423215702,
          0.022843626053767052,
          0.036650214376914056,
          0.063586307175310164,
          0.014163416620215776,
          0.30891492427180156,
          0.029130680538454128,
          0.012589518253728777,
          0.020846859037608637,
          0.023487244120678617,
          0.011058008046484179,
          0.023606814700297071,
          0.036951220308364015,
          0.02570899277154377,
          0.060072875576673034,
          0.028378823857736237,
          0.041642110377213812,
          0.027657243467982748,
          0.048365401776266831,
          0.038166380743738709,
          0.010365758979975024,
          0.091106736429473631
    ]
    train_test_div = np.array(train_test_div)

    #
    x = 4*np.arange(len(positive_label_perc))
    width = 0.7
    e = 1e-2
    plots_list = []
    #
    shift = 0
    curr_plot = plt.bar(x, positive_label_perc, color='k', width=width, label='positive label ratio')
    plots_list.append(curr_plot)
    #
    # shift += width
    # curr_plot = plt.bar(x+shift, train_test_div, color='gray', width=width, label='train-test divergence')
    # plots_list.append(curr_plot)
    #
    shift += width
    curr_plot = plt.bar(x+shift, org_max_lkl_amr_only+e, color='c', width=width, label='org max lkl (amr)')
    plots_list.append(curr_plot)
    #
    # shift += width
    # curr_plot = plt.bar(x+shift, org_max_lkl_sdg_only+e, color='m', width=width, label='org max lkl (sdg)')
    # plots_list.append(curr_plot)
    #
    # shift += width
    # curr_plot = plt.bar(x+shift, org_max_lkl_amr_sdg_zero+e, color='seashell', width=width, label='max lkl (amr,sdg)')
    # plots_list.append(curr_plot)
    #
    # shift += width
    # curr_plot = plt.bar(x+shift, org_max_lkl+e, color='r', width=width, label='max lkl (amr+sdg)')
    # plots_list.append(curr_plot)
    #
    shift += width
    curr_plot = plt.bar(x+shift, mmd_amr+e, color='b', width=width, label='mmd (amr)')
    plots_list.append(curr_plot)
    #
    shift += width
    curr_plot = plt.bar(x+shift, kl_kd_amr+e, color='r', width=width, label='KL-D (amr)')
    plots_list.append(curr_plot)
    #
    # shift += width
    # curr_plot = plt.bar(x+shift, mmd_sdg+e, color='yellowgreen', width=width, label='mmd (sdg)')
    # plots_list.append(curr_plot)
    #
    # shift += width
    # curr_plot = plt.bar(x+shift, mmd+e, color='g', width=width, label='mmd (amr+sdg)')
    # plots_list.append(curr_plot)
    #
    # shift += width
    # curr_plot = plt.bar(x+shift, mmd_amr_sdg_zero+e, color='orange', width=width, label='mmd (amr,sdg)')
    # plots_list.append(curr_plot)
    #
    plt.ylabel('F1 Score')
    plt.xlabel('PubMed Articles on Cancer Research')
    #
    plt.legend(handles=plots_list, loc=2, prop={'size':6})
    #
    plt.savefig('./validation_plots_nd_text_output/summary/amr_sdg_f1_train_fixed.pdf', dpi=300, format='pdf')
    #
    #
    plt.close()
    # #
    sort_idx = train_test_div.argsort()
    plt.plot(train_test_div[sort_idx], mmd_amr[sort_idx]/(org_max_lkl_amr_only[sort_idx]+1e-2), 'bx-', label='mmd(amr) / max_lkl(amr)')
    plt.plot(train_test_div[sort_idx], kl_kd_amr[sort_idx]/(org_max_lkl_amr_only[sort_idx]+1e-2), 'rx-', label='KL-D(amr) / max_lkl(amr)')
    plt.legend(loc=2, prop={'size':6})
    plt.xlabel('Train-Test Maximum Mean Discrepancy')
    plt.ylabel('F1 Score')
    plt.savefig('./validation_plots_nd_text_output/summary/amr_sdg_f1_train_test_div_train_fixed.pdf', dpi=300, format='pdf')
