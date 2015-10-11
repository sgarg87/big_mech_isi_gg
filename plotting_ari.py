import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    org_max_lkl_amr_only = [
               0.17407454230735225,
               0.19425243964639896,
               0.13758268824367767,
               0.18627857585132396,
               0.20425787422190944,
               0.02823747251475521,
               0.3103330476443522,
               0.18449750583181718,
               0.29032554691280155,
               0.232737599072058,
               0.2502251576103273
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
               0.12725462104319257,
               0.35120660939857046,
               0.2273176267837996,
               0.22630449002345374,
               0.10181762582027494,
               0.11374333228741795,
               0.3085843320924197,
               0.15486208673990698,
               0.23329988610178837,
               0.31633524721259887,
               0.7608903781713737
          ]
    mmd_amr = np.array(mmd_amr)
    #
    mmd_sdg = [
          ]
    mmd_sdg = np.array(mmd_sdg)
    #

    kl_kd_amr = [
               0.11417418032279479,
               0.24430246202809994,
               0.3805106098045751,
               0.16649807487258353,
               0.14637112161324872,
               0.14988502965024786,
               0.3169489676090279,
               0.18473359644844414,
               0.3175111210624373,
               0.22741607130721875,
               0.7608903781713737
          ]
    kl_kd_amr = np.array(kl_kd_amr)

    positive_label_perc = [
               0.069731800766283519,
               0.104,
               0.028571428571428571,
               0.099378881987577633,
               0.14476284584980237,
               0.015873015873015872,
               0.052054794520547946,
               0.056364617044228696,
               0.075971731448763249,
               0.030227948463825569,
               0.054054054054054057
          ]

    positive_label_perc = np.array(positive_label_perc)

    #
    train_test_div = [
          0.0049736208967558039,
          0.025688560967480914,
          0.021061710993721824,
          0.012387534290935361,
          0.0036165764787032557,
          0.047419643422435175,
          0.011508591576387617,
          0.0030118924598081902,
          0.011328775204574265,
          0.0070204800001698415,
          0.088407416092293828
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
    plt.ylabel('Adjusted Rand Index')
    plt.xlabel('PubMed Articles on Cancer Research')
    #
    plt.legend(handles=plots_list, loc=2, prop={'size':6})
    #
    plt.savefig('./validation_plots_nd_text_output/summary/amr_sdg_ari.pdf', dpi=300, format='pdf')
    #
    #
    plt.close()
    # #
    sort_idx = train_test_div.argsort()
    plt.plot(train_test_div[sort_idx], mmd_amr[sort_idx]/org_max_lkl_amr_only[sort_idx], 'bx-', label='mmd(amr) / max_lkl(amr)')
    plt.hold(True)
    plt.plot(train_test_div[sort_idx], kl_kd_amr[sort_idx]/org_max_lkl_amr_only[sort_idx], 'rx-', label='KL-D(amr) / max_lkl(amr)')
    plt.legend(loc=2, prop={'size':6})
    plt.xlabel('Train-Test Maximum Mean Discrepancy')
    plt.ylabel('Adjusted Rand Index')
    plt.savefig('./validation_plots_nd_text_output/summary/amr_sdg_ari_train_test_div.pdf', dpi=300, format='pdf')
