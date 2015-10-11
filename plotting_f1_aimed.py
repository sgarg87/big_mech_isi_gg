import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    org_amr = [
               [
                    0.4444444444444444,
                    0.35200000000000004,
                    0.26666666666666666,
                    0.23312883435582818,
                    0.3870967741935484
               ],
               [
                    0.35620915032679734,
                    0.3309608540925267,
                    0.34931506849315064,
                    0.32881355932203393,
                    0.3903225806451613
               ],
               [
                    0.37237237237237236,
                    0.33571428571428574,
                    0.3722627737226277,
                    0.3972602739726027,
                    0.32098765432098764
               ],
               [
                    0.4615384615384615,
                    0.4748603351955307,
                    0.4722222222222222,
                    0.4507936507936508,
                    0.43243243243243246
               ],
               [
                    0.45009784735812136,
                    0.46696035242290745,
                    0.37762237762237766,
                    0.39540229885057465,
                    0.37401574803149606
               ],
               [
                    0.36990595611285265,
                    0.36619718309859156,
                    0.4491228070175438,
                    0.40449438202247195,
                    0.39867109634551495
               ],
               [
                    0.32558139534883723,
                    0.411764705882353,
                    0.32441471571906355,
                    0.3545454545454545,
                    0.3542435424354244
               ],
               [
                    0.3797856049004594,
                    0.3390357698289269,
                    0.3412162162162162,
                    0.392330383480826,
                    0.37217391304347824
               ],
               [
                    0.3960720130932897,
                    0.4364820846905537,
                    0.4236453201970444,
                    0.4013961605584642,
                    0.4012251148545177
               ],
               [
                    0.48739495798319327,
                    0.488,
                    0.36080178173719374,
                    0.3457446808510638,
                    0.44938271604938274
               ]
          ]
    org_amr = np.array(org_amr)
    print 'org_amr', org_amr.std()

    org_sdg = [
               [
                    0.46846846846846846,
                    0.5673758865248227,
                    0.5179856115107914,
                    0.5573770491803278,
                    0.5393258426966292
               ],
               [
                    0.2157676348547718,
                    0.15044247787610618,
                    0.13973799126637554,
                    0.18852459016393444,
                    0.16666666666666669
               ],
               [
                    0.25675675675675674,
                    0.17490494296577946,
                    0.1702127659574468,
                    0.16608996539792384,
                    0.10612244897959183
               ],
               [
                    0.2716049382716049,
                    0.3,
                    0.36774193548387096,
                    0.3028391167192429,
                    0.22545454545454544
               ],
               [
                    0.17964071856287425,
                    0.24468085106382978,
                    0.12987012987012989,
                    0.138328530259366,
                    0.2558746736292428
               ],
               [
                    0.34602076124567477,
                    0.3230240549828179,
                    0.34983498349834985,
                    0.33098591549295775,
                    0.3076923076923077
               ],
               [
                    0.4411085450346421,
                    0.44067796610169496,
                    0.3163265306122449,
                    0.2914798206278027,
                    0.27999999999999997
               ],
               [
                    0.24896265560165975,
                    0.22784810126582278,
                    0.14285714285714285,
                    0.25357873210633947,
                    0.23776223776223773
               ],
               [
                    0.2657952069716775,
                    0.1728395061728395,
                    0.2372093023255814,
                    0.24369747899159663,
                    0.3246753246753247
               ],
               [
                    0.2014388489208633,
                    0.20408163265306123,
                    0.24615384615384617,
                    0.12794612794612795,
                    0.15894039735099338
               ]
          ]
    org_sdg = np.array(org_sdg)
    print 'org_sdg', org_sdg.std()

    org = [
               [
                    0.4089456869009584,
                    0.4090909090909091,
                    0.4742268041237113,
                    0.3684210526315789,
                    0.3607843137254902
               ],
               [
                    0.27727645611156687,
                    0.34540859309182814,
                    0.33399602385685884,
                    0.28053435114503816,
                    0.32021466905187834
               ],
               [
                    0.42089552238805966,
                    0.36986301369863017,
                    0.3499999999999999,
                    0.37577639751552794,
                    0.32692307692307687
               ],
               [
                    0.3427561837455831,
                    0.4029850746268657,
                    0.4096045197740113,
                    0.41002949852507375,
                    0.36500754147812975
               ],
               [
                    0.3073639274279616,
                    0.3400503778337532,
                    0.39333333333333337,
                    0.3337453646477132,
                    0.37727759914255093
               ],
               [
                    0.33943427620632277,
                    0.3716814159292035,
                    0.38765008576329324,
                    0.38376383763837635,
                    0.427652733118971
               ],
               [
                    0.42477876106194684,
                    0.3225190839694656,
                    0.29559748427672955,
                    0.3654545454545454,
                    0.3793103448275862
               ],
               [
                    0.3583333333333334,
                    0.3213988343047461,
                    0.31438721136767317,
                    0.3193717277486911,
                    0.3566666666666667
               ],
               [
                    0.35078053259871445,
                    0.3324048282265552,
                    0.31568016614745587,
                    0.3306930693069307,
                    0.3238095238095238
               ],
               [
                    0.35750000000000004,
                    0.3570566948130277,
                    0.34403080872913994,
                    0.3565323565323565,
                    0.3916768665850673
               ]
          ]
    org = np.array(org)
    print 'org', org.std()

    org_max_lkl_amr_only = [
               [
                    0.31034482758620696,
                    0.2173913043478261,
                    0.20833333333333334,
                    0.21875000000000003,
                    0.2978723404255319
               ],
               [
                    0.5605095541401274,
                    0.5616438356164384,
                    0.5599999999999999,
                    0.54421768707483,
                    0.65
               ],
               [
                    0.5185185185185186,
                    0.4571428571428571,
                    0.5263157894736842,
                    0.5277777777777778,
                    0.4347826086956522
               ],
               [
                    0.6,
                    0.5416666666666667,
                    0.5631067961165048,
                    0.5609756097560975,
                    0.5274725274725275
               ],
               [
                    0.6333333333333334,
                    0.6238532110091743,
                    0.616822429906542,
                    0.5904761904761904,
                    0.5641025641025642
               ],
               [
                    0.5909090909090909,
                    0.5609756097560976,
                    0.6172839506172839,
                    0.6024096385542169,
                    0.6190476190476191
               ],
               [
                    0.4052287581699347,
                    0.5256410256410257,
                    0.4430379746835443,
                    0.4311377245508982,
                    0.4342105263157895
               ],
               [
                    0.4729729729729729,
                    0.45637583892617445,
                    0.4316546762589928,
                    0.4871794871794873,
                    0.4657534246575343
               ],
               [
                    0.524390243902439,
                    0.5276073619631901,
                    0.47435897435897434,
                    0.4840764331210191,
                    0.5212121212121212
               ],
               [
                    0.6504065040650406,
                    0.6029411764705882,
                    0.5,
                    0.4423076923076923,
                    0.5945945945945945
               ]
          ]

    org_max_lkl_amr_only = np.array(org_max_lkl_amr_only)
    print 'org_max_lkl_amr_only.std()', org_max_lkl_amr_only.std()

    org_max_lkl_sdg_only = [
               [
                    0.4680851063829787,
                    0.6071428571428572,
                    0.4912280701754386,
                    0.5957446808510638,
                    0.6285714285714286
               ],
               [
                    0.36666666666666664,
                    0.2803738317757009,
                    0.2702702702702703,
                    0.3220338983050848,
                    0.29310344827586204
               ],
               [
                    0.37333333333333335,
                    0.30303030303030304,
                    0.30985915492957744,
                    0.30985915492957744,
                    0.22580645161290325
               ],
               [
                    0.32989690721649484,
                    0.35999999999999993,
                    0.46,
                    0.39583333333333326,
                    0.2682926829268293
               ],
               [
                    0.3058823529411765,
                    0.4040404040404041,
                    0.25,
                    0.2619047619047619,
                    0.44897959183673475
               ],
               [
                    0.5063291139240506,
                    0.430379746835443,
                    0.5,
                    0.47500000000000003,
                    0.38888888888888895
               ],
               [
                    0.4779874213836478,
                    0.4727272727272728,
                    0.4736842105263158,
                    0.4545454545454546,
                    0.391304347826087
               ],
               [
                    0.490566037735849,
                    0.42718446601941745,
                    0.3181818181818182,
                    0.47787610619469023,
                    0.4313725490196078
               ],
               [
                    0.3770491803278688,
                    0.2772277227722772,
                    0.37383177570093457,
                    0.35772357723577236,
                    0.4186046511627907
               ],
               [
                    0.41269841269841273,
                    0.3529411764705882,
                    0.4057971014492754,
                    0.3448275862068965,
                    0.34375
               ]
          ]

    org_max_lkl_sdg_only = np.array(org_max_lkl_sdg_only)
    print 'org_max_lkl_sdg_only.std()', org_max_lkl_sdg_only.std()


    org_max_lkl = [
               [
                    0.3132530120481928,
                    0.3125,
                    0.411764705882353,
                    0.3611111111111111,
                    0.3823529411764706
               ],
               [
                    0.5957446808510638,
                    0.6382978723404255,
                    0.5780346820809249,
                    0.5833333333333334,
                    0.6333333333333333
               ],
               [
                    0.6250000000000001,
                    0.5391304347826086,
                    0.5686274509803921,
                    0.5490196078431372,
                    0.5263157894736842
               ],
               [
                    0.509090909090909,
                    0.5528455284552845,
                    0.5426356589147286,
                    0.56,
                    0.5299145299145299
               ],
               [
                    0.6511627906976744,
                    0.619047619047619,
                    0.676470588235294,
                    0.6461538461538462,
                    0.6619718309859154
               ],
               [
                    0.5961538461538461,
                    0.6285714285714286,
                    0.6538461538461537,
                    0.66,
                    0.6595744680851063
               ],
               [
                    0.5,
                    0.451219512195122,
                    0.44444444444444453,
                    0.47311827956989244,
                    0.459016393442623
               ],
               [
                    0.5664739884393064,
                    0.5632183908045978,
                    0.548780487804878,
                    0.5185185185185185,
                    0.5714285714285715
               ],
               [
                    0.5222222222222223,
                    0.5263157894736843,
                    0.4880952380952381,
                    0.5027932960893856,
                    0.5027322404371585
               ],
               [
                    0.5833333333333334,
                    0.6086956521739131,
                    0.6000000000000001,
                    0.5970149253731343,
                    0.6131386861313869
               ]
          ]


    org_max_lkl = np.array(org_max_lkl)
    print 'org_max_lkl.std()', org_max_lkl.std()
    #

    mmd = [
               [
                    0.2857142857142857,
                    0.3111111111111111,
                    0.34782608695652173,
                    0.25000000000000006,
                    0.29268292682926833
               ],
               [
                    0.6700507614213198,
                    0.6918918918918919,
                    0.6477272727272727,
                    0.6703296703296703,
                    0.6629834254143647
               ],
               [
                    0.5523809523809524,
                    0.5871559633027522,
                    0.5631067961165048,
                    0.5523809523809524,
                    0.5399999999999999
               ],
               [
                    0.4273504273504274,
                    0.48437500000000006,
                    0.5039370078740157,
                    0.4878048780487805,
                    0.4715447154471545
               ],
               [
                    0.6382978723404256,
                    0.573529411764706,
                    0.5693430656934306,
                    0.5925925925925926,
                    0.5972222222222222
               ],
               [
                    0.5544554455445545,
                    0.5252525252525253,
                    0.5576923076923077,
                    0.5306122448979592,
                    0.5631067961165049
               ],
               [
                    0.4745762711864407,
                    0.4861878453038674,
                    0.4555555555555556,
                    0.4888888888888888,
                    0.47398843930635837
               ],
               [
                    0.5082872928176796,
                    0.48648648648648646,
                    0.5,
                    0.5,
                    0.5333333333333334
               ],
               [
                    0.5333333333333333,
                    0.5664739884393063,
                    0.5308641975308641,
                    0.5365853658536585,
                    0.5476190476190476
               ],
               [
                    0.5030674846625768,
                    0.5,
                    0.5149700598802396,
                    0.5030674846625767,
                    0.47852760736196326
               ]
          ]
    mmd = np.array(mmd)
    print 'mmd', mmd.std()

    mmd_amr = [
               [
                    0.30769230769230765,
                    0.12903225806451613,
                    0.24000000000000002,
                    0.3243243243243243,
                    0.2941176470588235
               ],
               [
                    0.6524064171122995,
                    0.6853932584269663,
                    0.6630434782608695,
                    0.6594594594594595,
                    0.6910994764397906
               ],
               [
                    0.5436893203883495,
                    0.4705882352941176,
                    0.49462365591397855,
                    0.5306122448979592,
                    0.43373493975903615
               ],
               [
                    0.47540983606557374,
                    0.4793388429752067,
                    0.5,
                    0.504201680672269,
                    0.45901639344262296
               ],
               [
                    0.5999999999999999,
                    0.5954198473282442,
                    0.5735294117647058,
                    0.5619834710743802,
                    0.5925925925925926
               ],
               [
                    0.6262626262626264,
                    0.5473684210526316,
                    0.5306122448979591,
                    0.5274725274725274,
                    0.56
               ],
               [
                    0.4431818181818182,
                    0.5172413793103449,
                    0.4945054945054945,
                    0.5168539325842696,
                    0.47337278106508873
               ],
               [
                    0.5119047619047619,
                    0.49142857142857144,
                    0.49101796407185627,
                    0.5844155844155845,
                    0.4615384615384615
               ],
               [
                    0.5609756097560976,
                    0.5802469135802469,
                    0.5185185185185186,
                    0.546583850931677,
                    0.576271186440678
               ],
               [
                    0.4968944099378882,
                    0.5116279069767442,
                    0.5086705202312138,
                    0.4691358024691359,
                    0.44999999999999996
               ]
          ]
    #
    mmd_amr = np.array(mmd_amr)
    print 'mmd_amr', mmd_amr.std()
    #
    mmd_sdg = [
               [
                    0.31999999999999995,
                    0.3703703703703704,
                    0.3272727272727273,
                    0.3508771929824561,
                    0.3018867924528302
               ],
               [
                    0.6709677419354839,
                    0.6114649681528662,
                    0.6114649681528662,
                    0.6219512195121951,
                    0.6499999999999999
               ],
               [
                    0.43137254901960786,
                    0.47058823529411764,
                    0.4660194174757281,
                    0.5046728971962617,
                    0.4444444444444445
               ],
               [
                    0.4833333333333334,
                    0.43859649122807015,
                    0.4601769911504424,
                    0.4655172413793103,
                    0.4521739130434783
               ],
               [
                    0.5074626865671642,
                    0.5147058823529411,
                    0.46616541353383456,
                    0.4852941176470588,
                    0.5294117647058824
               ],
               [
                    0.5,
                    0.5098039215686274,
                    0.5333333333333333,
                    0.46938775510204084,
                    0.48484848484848486
               ],
               [
                    0.3731343283582089,
                    0.35658914728682173,
                    0.328125,
                    0.3816793893129771,
                    0.35555555555555557
               ],
               [
                    0.5257142857142858,
                    0.5027932960893855,
                    0.5,
                    0.5222222222222221,
                    0.4678362573099415
               ],
               [
                    0.4805194805194805,
                    0.5,
                    0.49664429530201337,
                    0.5064935064935066,
                    0.4675324675324676
               ],
               [
                    0.4275862068965517,
                    0.4666666666666667,
                    0.44594594594594594,
                    0.47058823529411775,
                    0.47058823529411753
               ]
          ]
    mmd_sdg = np.array(mmd_sdg)
    print 'mmd_sdg', mmd_sdg.std()
    #
    positive_label_perc = [
               [
                    0.08383233532934131,
                    0.09580838323353294,
                    0.08982035928143713,
                    0.08982035928143713,
                    0.08383233532934131
               ],
               [
                    0.5602409638554217,
                    0.5240963855421686,
                    0.5,
                    0.5060240963855421,
                    0.5240963855421686
               ],
               [
                    0.4117647058823529,
                    0.45098039215686275,
                    0.4117647058823529,
                    0.4215686274509804,
                    0.4117647058823529
               ],
               [
                    0.296551724137931,
                    0.33793103448275863,
                    0.33793103448275863,
                    0.3448275862068966,
                    0.3310344827586207
               ],
               [
                    0.42857142857142855,
                    0.41496598639455784,
                    0.40816326530612246,
                    0.4217687074829932,
                    0.43537414965986393
               ],
               [
                    0.4574468085106383,
                    0.43617021276595747,
                    0.44680851063829785,
                    0.44680851063829785,
                    0.4148936170212766
               ],
               [
                    0.31176470588235294,
                    0.3,
                    0.3,
                    0.31176470588235294,
                    0.28823529411764703
               ],
               [
                    0.34054054054054056,
                    0.35135135135135137,
                    0.34054054054054056,
                    0.32432432432432434,
                    0.33513513513513515
               ],
               [
                    0.4451219512195122,
                    0.4451219512195122,
                    0.42073170731707316,
                    0.4573170731707317,
                    0.45121951219512196
               ],
               [
                    0.3698630136986301,
                    0.3698630136986301,
                    0.3972602739726027,
                    0.3767123287671233,
                    0.3561643835616438
               ]
          ]
    #
    positive_label_perc = np.array(positive_label_perc)
    print 'positive_label_perc.mean()', positive_label_perc.mean()
    print 'positive_label_perc.std()', positive_label_perc.std()

    #
    train_test_div_amr = [
          [
               0.02504586521536112,
               0.025466675870120525,
               0.02996022067964077,
               0.027385313995182514,
               0.022743869572877884
          ],
          [
               0.008345676120370626,
               0.00928345462307334,
               0.008787323255091906,
               0.008503599558025599,
               0.008540398441255093
          ],
          [
               0.007846524473279715,
               0.007818245328962803,
               0.008844217285513878,
               0.008996722754091024,
               0.008663162123411894
          ],
          [
               0.006979251280426979,
               0.006940536201000214,
               0.007296127267181873,
               0.007259749807417393,
               0.007153334561735392
          ],
          [
               0.007210263982415199,
               0.0069225807674229145,
               0.006992215756326914,
               0.007378256414085627,
               0.006538312882184982
          ],
          [
               0.008069033734500408,
               0.008355737663805485,
               0.008442413061857224,
               0.008847740478813648,
               0.008403029292821884
          ],
          [
               0.005538934841752052,
               0.00545234140008688,
               0.0053175101056694984,
               0.00514104263857007,
               0.00530257960781455
          ],
          [
               0.005903820041567087,
               0.005832941271364689,
               0.006031914614140987,
               0.006005541887134314,
               0.0064270105212926865
          ],
          [
               0.005899474024772644,
               0.005684267729520798,
               0.006021909415721893,
               0.00590129941701889,
               0.006123834289610386
          ],
          [
               0.006167404353618622,
               0.006357352249324322,
               0.006126000080257654,
               0.006296201143413782,
               0.0069918083027005196
          ]
     ]
    train_test_div_amr = np.array(train_test_div_amr)

    train_test_div_sdg = [
          [
               0.01548299059504643,
               0.014990547904744744,
               0.014293120359070599,
               0.015148942184168845,
               0.014480397570878267
          ],
          [
               0.009111619903706014,
               0.008312673948239535,
               0.008879032568074763,
               0.008309486729558557,
               0.008730242552701384
          ],
          [
               0.007617781084263697,
               0.006980111764278263,
               0.007572901668027043,
               0.007299387420061976,
               0.007919604249764234
          ],
          [
               0.007610911037772894,
               0.0075758451130241156,
               0.007796322635840625,
               0.007363534590695053,
               0.007909053470939398
          ],
          [
               0.006163638056023046,
               0.006320867134490982,
               0.006364472181303427,
               0.006078068079659715,
               0.00675217539537698
          ],
          [
               0.007886788051109761,
               0.008168362255673856,
               0.007462997396942228,
               0.00792049802839756,
               0.008304566552396864
          ],
          [
               0.00965226293192245,
               0.010116111399838701,
               0.008370586816454306,
               0.01208933282759972,
               0.01126286352518946
          ],
          [
               0.005633397318888456,
               0.005676706205122173,
               0.005659839604049921,
               0.005547529435716569,
               0.005701981019228697
          ],
          [
               0.00620895269094035,
               0.005869322514627129,
               0.006230761995539069,
               0.005983353476040065,
               0.005942653107922524
          ],
          [
               0.006411881477106363,
               0.006909890362294391,
               0.006602810055483133,
               0.006579178036190569,
               0.006403440493158996
          ]
     ]
    train_test_div_sdg = np.array(train_test_div_sdg)

    train_test_div = [
          [
               0.010947533417493105,
               0.011477824067696929,
               0.010777808958664536,
               0.011581227881833911,
               0.012664595851674676
          ],
          [
               0.004711986519396305,
               0.004555943887680769,
               0.004613109631463885,
               0.00469734356738627,
               0.004374326905235648
          ],
          [
               0.0042614813428372145,
               0.004408924374729395,
               0.0043069543316960335,
               0.004479952622205019,
               0.00423513213172555
          ],
          [
               0.004169645952060819,
               0.003985619870945811,
               0.003869002917781472,
               0.003918495262041688,
               0.003892062231898308
          ],
          [
               0.0036842739209532738,
               0.003636204404756427,
               0.00354440463706851,
               0.003733301069587469,
               0.003470398485660553
          ],
          [
               0.004170988453552127,
               0.004688509972766042,
               0.004566496703773737,
               0.004056194331496954,
               0.004458493087440729
          ],
          [
               0.003885598387569189,
               0.004441665718331933,
               0.004155049799010158,
               0.004458751762285829,
               0.0038189925253391266
          ],
          [
               0.003063423791900277,
               0.0031061810441315174,
               0.003205207409337163,
               0.0032424170058220625,
               0.003190702060237527
          ],
          [
               0.0030296139884740114,
               0.0030629385728389025,
               0.0031112583819776773,
               0.003055299399420619,
               0.0030715034808963537
          ],
          [
               0.0037317799869924784,
               0.003490648465231061,
               0.0033893233630806208,
               0.0033895375672727823,
               0.0035638867411762476
          ]
     ]
    train_test_div = np.array(train_test_div)
    print 'train_test_div.mean()', train_test_div.mean()
    print 'train_test_div.std()', train_test_div.std()
    #
    n = org_max_lkl.shape[0]
    x = 3*np.arange(n)
    width = 0.8
    e = 1e-3
    #
    font_size = 18
    plots_list = []
    shift = 0
    plt.ylim(0, 1)
    curr_plot = plt.bar(x+shift, org_max_lkl_sdg_only.mean(1), color='#FF70AA', hatch="\\", width=width, label='SDG (MSI)', yerr=org_max_lkl_sdg_only.std(1), ecolor='k')
    plots_list.append(curr_plot)
    shift += width
    curr_plot = plt.bar(x+shift, org_max_lkl_amr_only.mean(1), color='#FFFF66', hatch="/", width=width, label='AMR (MSI)', yerr=org_max_lkl_amr_only.std(1), ecolor='k')
    plots_list.append(curr_plot)
    shift += width
    curr_plot = plt.bar(x+shift, org_max_lkl.mean(1), color='#00FF00', hatch="\/", width=width, label='AMR-SDG (MSI)', yerr=org_max_lkl.std(1), ecolor='k')
    plots_list.append(curr_plot)
    # shift += width
    # curr_plot = plt.bar(x+shift, train_test_div.mean(1), color='k', width=width, label='Train-test div.', yerr=train_test_div.std(1))
    # plots_list.append(curr_plot)
    # shift += width
    # curr_plot = plt.bar(x+shift, positive_label_perc.mean(1), color='brown', width=width, label='Positive labels ratio', yerr=positive_label_perc.std(1))
    # plots_list.append(curr_plot)
    plt.ylabel('F1 Score', fontsize=font_size)
    plt.xlabel('AIMed Papers Abstract Sets', fontsize=font_size)
    # plt.title('F1 Score Comparison for Our Algorithm')
    sets_list = range(1,11)
    plt.xticks(x, sets_list, size=font_size)
    plt.yticks(size=font_size)
    plt.legend(handles=plots_list, loc=2, prop={'size': font_size})
    plt.savefig('./validation_plots_nd_text_output/summary/org_max_lkl_mmd_amr_vs_sdg_f1.pdf', dpi=300, format='pdf')
    plt.close()
