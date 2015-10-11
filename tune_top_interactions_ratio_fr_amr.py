import csv
import math
import numpy as np
import matplotlib.pyplot as plt


amr = 'amr'
lkl = 'lkl'
correct = 'correct'
sentence = 'sentence'


class TuneInteractionsRatio:
    def __init__(self):
        self.org_ratio = 0.66
        print 'assuming that the file is sorted by amr ascending and then by lkl descending.'
        self.amr_map = {}
        with open('./lkl_correction_list.csv', 'rU') as f:
            reader = csv.DictReader(f)
            for row in reader:
                curr_amr = row[amr]
                if curr_amr not in self.amr_map:
                    self.amr_map[curr_amr] = {}
                    self.amr_map[curr_amr][lkl] = []
                    self.amr_map[curr_amr][correct] = []
                self.amr_map[curr_amr][lkl].append(float(row[lkl]))
                self.amr_map[curr_amr][correct].append(bool(int(row[correct])))
        print self.amr_map

    def eval_mean_performance(self, top_ratio):
        total_correct = 0
        total_selected = 0
        for curr_amr in self.amr_map:
            curr_lkl_list = self.amr_map[curr_amr][lkl]
            min_idx = int(math.ceil(len(curr_lkl_list)*top_ratio))
            curr_num_correct = sum(self.amr_map[curr_amr][correct][0:min_idx])
            total_selected += min_idx
            total_correct += curr_num_correct
        return total_correct, total_selected

    def tune(self):
        top_ratio_actual_list = []
        precision_list = []
        throughput_list = []
        print '{}:{}'.format('top_ratio', 'precision', 'throughput')
        for curr_top_ratio in np.arange(0.1, 1, 0.03):
            curr_total_correct, curr_total_selected = self.eval_mean_performance(curr_top_ratio)
            #
            curr_top_ratio_actual = curr_top_ratio*self.org_ratio
            curr_precision = curr_total_correct/float(curr_total_selected)
            #
            top_ratio_actual_list.append(curr_top_ratio_actual)
            precision_list.append(curr_precision)
            throughput_list.append(curr_total_correct)
            #
            print '{}:{}:{}'.format(curr_top_ratio_actual, curr_precision, curr_total_correct)
        plt.subplot(211)
        plt.plot(top_ratio_actual_list, precision_list, 'kx-')
        plt.xlabel('Ratio of top lkl')
        plt.ylabel('precision')
        plt.subplot(212)
        plt.plot(top_ratio_actual_list, throughput_list, 'ro-')
        plt.xlabel('Ratio of top lkl')
        plt.ylabel('throughput (correct)')
        plt.savefig('./tuning.pdf', format='pdf', dpi=300)
        plt.close()


if __name__ == '__main__':
    tir_obj = TuneInteractionsRatio()
    tir_obj.tune()

