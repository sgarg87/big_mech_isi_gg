import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import constants_darpa_json_format as cdjf


scores_fr_unevaluated = {'S-Value', 'S-BG', 'R'}
col_info_extracted_score = 'Info. Extracted Score'
col_card = 'Card'
col_model_relation_score = 'Model Relation Score'

results_file = './mitre_june_2015_eval_results/results_machine_ours.csv'

is_filtered = False
sparse_gap = 1

def get_evaluated_results_data(file_path):
    evaluated_results_list = []
    with open(file_path, 'rU') as csv_file:
        reader = csv.DictReader(csv_file)
        for curr_row in reader:
            if curr_row[col_info_extracted_score] not in scores_fr_unevaluated:
                evaluated_results_list.append(curr_row)
                print curr_row[col_info_extracted_score]
    print len(evaluated_results_list)
    return evaluated_results_list


def get_weight_score_card_map(results_list):
    weight_score_map = {}
    weight_score_map['weight'] = []
    weight_score_map['score'] = []
    weight_score_map['info_relation_avg_score'] = []
    weight_score_map['weight_fr_model_relation'] = []
    weight_score_map['model_relation_score'] = []
    for curr_result_map in results_list:
        curr_card_path = './mitre_june_2015_eval_results/'+curr_result_map[col_card]
        with open(curr_card_path, 'r') as curr_card_file:
            card_json = json.load(curr_card_file)
            curr_weight = float(card_json['weight'])
            weight_score_map['weight'].append(curr_weight)
            curr_score = float(curr_result_map[col_info_extracted_score])
            weight_score_map['score'].append(curr_score)
            if curr_score > 0:
                curr_model_relation_score = float(curr_result_map[col_model_relation_score])
                curr_avg_score = float(curr_score+curr_model_relation_score)/2
                weight_score_map['info_relation_avg_score'].append(curr_avg_score)
                #
                weight_score_map['weight_fr_model_relation'].append(curr_weight)
                weight_score_map['model_relation_score'].append(curr_model_relation_score)
            else:
                weight_score_map['info_relation_avg_score'].append(curr_score)
    return weight_score_map


def get_num_cards_not_related_to_model_by_grounding(results_list, identifiers_list):
    def is_identifier_in_model(id):
        # if 'mouse' in id.lower():
        #     return False
        if id not in identifiers_list:
            return False
        return True
    count = 0
    invalid_card_paths_list = []
    scores = []
    for curr_result_map in results_list:
        curr_card_path = './mitre_june_2015_eval_results/'+curr_result_map[col_card]
        with open(curr_card_path, 'r') as curr_card_file:
            card_json = json.load(curr_card_file)
        pa_in_model = True
        pb_in_model = True
        p_a = card_json[cdjf.extracted_information][cdjf.participant_a]
        p_b = card_json[cdjf.extracted_information][cdjf.participant_b]
        if cdjf.entities not in p_b:
            pb_in_model = is_identifier_in_model(p_b[cdjf.identifier]) and p_b[cdjf.in_model]
        else:
            pb_in_model = False
            for curr_entity in p_b[cdjf.entities]:
                pb_in_model = pb_in_model or (is_identifier_in_model(curr_entity[cdjf.identifier]) and curr_entity[cdjf.in_model])
        #
        if p_a is not None:
            pa_in_model = is_identifier_in_model(p_a[cdjf.identifier]) and p_a[cdjf.in_model]
        if not (pa_in_model or pb_in_model):
            invalid_card_paths_list.append(curr_result_map[col_card])
            scores.append(curr_result_map[col_info_extracted_score])
            continue
    with open('./mitre_june_2015_eval_results/cards_not_related_to_model_by_grounding.txt', 'w') as count_file:
        count_file.write('count: ' + str(len(invalid_card_paths_list)))
        count_file.write('\n')
        count_file.write('card paths are listed here')
        count_file.write('\n')
        assert len(scores) == len(invalid_card_paths_list)
        for i in range(len(invalid_card_paths_list)):
            count_file.write(invalid_card_paths_list[i]+','+scores[i])
            count_file.write('\n')
    return invalid_card_paths_list


def get_identifiers_list():
    with open('../identifiers_list.json', 'r') as f:
        list1 = json.load(f)
    with open('../ras_2_gold_entities.json', 'r') as f:
        list2 = json.load(f)
    return list1+list2


def gen_precision_throughput_curve(org_weight_score_map, filtered_weight_score_map, file_path, is_strict):
    def sub_plot(weight_score_map, is_secondary=False):
        min_weight = min(weight_score_map['weight'])
        max_weight = max(weight_score_map['weight'])
        weights = np.array(weight_score_map['weight'])
        org_scores = np.array(weight_score_map['info_relation_avg_score'])
        weight_range = max_weight-min_weight
        weight_step = weight_range/float(30)
        print 'weight_step', weight_step
        soft_scores = []
        scores = []
        precisions = []
        soft_precisions = []
        weight_thresholds = []
        weights_sorted = np.copy(weights)
        weights_sorted.sort()
        for curr_weight_threshold in weights[0::sparse_gap]:
            print 'curr_weight_threshold', curr_weight_threshold
            #
            sel_weights_idx = np.where(weights >= curr_weight_threshold)
            curr_n = sel_weights_idx[0].size
            if curr_n <= 1:
                continue
            #
            weight_thresholds.append(curr_weight_threshold)
            sel_score = org_scores[sel_weights_idx]
            #
            # print 'sel_score', sel_score
            curr_score = sel_score.sum()
            scores.append(curr_score)
            #
            curr_soft_score = np.where(sel_score > 0)[0].size
            soft_scores.append(curr_soft_score)
            #
            curr_precision = curr_score/float(curr_n)
            precisions.append(curr_precision)
            #
            curr_soft_precision = curr_soft_score/float(curr_n)
            soft_precisions.append(curr_soft_precision)
            print 'curr_n', curr_n
            print 'curr_score', curr_score
            print 'curr_precision', curr_precision
            print 'curr_soft_score', curr_soft_score
            print 'curr_soft_precision', curr_soft_precision
            print '************************************'
        if not is_secondary:
            if is_strict:
                write_csv(soft_scores, precisions, file_path)
                plt.plot(soft_scores, precisions, 'bx', label='original')
            else:
                write_csv(soft_scores, soft_precisions, file_path)
                plt.plot(soft_scores, soft_precisions, 'bx', label='original')
        else:
            if is_strict:
                plt.plot(soft_scores, precisions, 'ro', label='filtered')
            else:
                plt.plot(soft_scores, soft_precisions, 'ro', label='filtered')
        if not is_secondary:
            plt.hold(True)
            plt.xlabel('No. of valid cards')
            if is_strict:
                plt.ylabel('Strict Precision')
            else:
                plt.ylabel('Generous Precision')
            plt.ylim(0, 1)

    plt.close()
    sub_plot(org_weight_score_map)
    if is_filtered:
        sub_plot(filtered_weight_score_map, is_secondary=True)
        plt.legend()
    if is_strict:
        plt.title('Strict Precision Curve')
    else:
        plt.title('Generous Precision Curve')
    plt.savefig(file_path+'.pdf', dpi=300, format='pdf')
    plt.close()


def gen_precision_relation_to_model_throughput_curve(org_weight_score_map, filtered_weight_score_map, file_path):
    def sub_plot(weight_score_map, is_secondary=False):
        min_weight = min(weight_score_map['weight_fr_model_relation'])
        max_weight = max(weight_score_map['weight_fr_model_relation'])
        weights = np.array(weight_score_map['weight_fr_model_relation'])
        org_scores = np.array(weight_score_map['model_relation_score'])
        weight_range = max_weight-min_weight
        weight_step = weight_range/float(30)
        print 'weight_step', weight_step
        soft_scores = []
        scores = []
        precisions = []
        soft_precisions = []
        weight_thresholds = []
        weights_sorted = np.copy(weights)
        weights_sorted.sort()
        for curr_weight_threshold in weights[0::sparse_gap]:
            print 'curr_weight_threshold', curr_weight_threshold
            #
            sel_weights_idx = np.where(weights >= curr_weight_threshold)
            curr_n = sel_weights_idx[0].size
            if curr_n <= 1:
                continue
            #
            weight_thresholds.append(curr_weight_threshold)
            sel_score = org_scores[sel_weights_idx]
            #
            # print 'sel_score', sel_score
            curr_score = sel_score.sum()
            scores.append(curr_score)
            #
            curr_soft_score = np.where(sel_score > 0)[0].size
            soft_scores.append(curr_soft_score)
            #
            curr_precision = curr_score/float(curr_n)
            precisions.append(curr_precision)
            #
            curr_soft_precision = curr_soft_score/float(curr_n)
            soft_precisions.append(curr_soft_precision)
            print 'curr_n', curr_n
            print 'curr_score', curr_score
            print 'curr_precision', curr_precision
            print 'curr_soft_score', curr_soft_score
            print 'curr_soft_precision', curr_soft_precision
            print '************************************'
        if not is_secondary:
            write_csv(soft_scores, precisions, file_path)
            plt.plot(soft_scores, precisions, 'bx', label='original')
        else:
            plt.plot(soft_scores, precisions, 'ro', label='filtered')
        if not is_secondary:
            plt.hold(True)
            plt.xlabel('No. of valid cards')
            plt.ylabel('Strict Precision (Model)')
            plt.ylim(0, 1)

    plt.close()
    sub_plot(org_weight_score_map)
    if is_filtered:
        sub_plot(filtered_weight_score_map, is_secondary=True)
        plt.legend(loc="lower right")
    plt.title('Relation To Model Strict Precision Curve')
    plt.savefig(file_path+'.pdf', dpi=300, format='pdf')
    plt.close()

def write_csv(col1, col2, filepath):
    with open(filepath+'.csv', 'w') as f:
        col_num_of_cards = 'no. of valid cards'
        col_precision = 'precision'
        column_names = [col_num_of_cards, col_precision]
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        assert len(col1) == len(col2)
        for i in range(len(col1)):
            writer.writerow({col_num_of_cards: col1[i], col_precision: col2[i]})


def gen_precision_info_throughput_curve(org_weight_score_map, filtered_weight_score_map, file_path):
    def sub_plot(weight_score_map, is_secondary=False):
        min_weight = min(weight_score_map['weight'])
        max_weight = max(weight_score_map['weight'])
        weights = np.array(weight_score_map['weight'])
        org_scores = np.array(weight_score_map['score'])
        weight_range = max_weight-min_weight
        weight_step = weight_range/float(30)
        print 'weight_step', weight_step
        soft_scores = []
        scores = []
        precisions = []
        soft_precisions = []
        weight_thresholds = []
        weights_sorted = np.copy(weights)
        weights_sorted.sort()
        for curr_weight_threshold in weights[0::sparse_gap]:
            print 'curr_weight_threshold', curr_weight_threshold
            #
            sel_weights_idx = np.where(weights >= curr_weight_threshold)
            curr_n = sel_weights_idx[0].size
            if curr_n <= 1:
                continue
            #
            weight_thresholds.append(curr_weight_threshold)
            sel_score = org_scores[sel_weights_idx]
            #
            # print 'sel_score', sel_score
            curr_score = sel_score.sum()
            scores.append(curr_score)
            #
            curr_soft_score = np.where(sel_score > 0)[0].size
            soft_scores.append(curr_soft_score)
            #
            curr_precision = curr_score/float(curr_n)
            precisions.append(curr_precision)
            #
            curr_soft_precision = curr_soft_score/float(curr_n)
            soft_precisions.append(curr_soft_precision)
            print 'curr_n', curr_n
            print 'curr_score', curr_score
            print 'curr_precision', curr_precision
            print 'curr_soft_score', curr_soft_score
            print 'curr_soft_precision', curr_soft_precision
            print '************************************'
        if not is_secondary:
            write_csv(soft_scores, precisions, file_path)
            plt.plot(soft_scores, precisions, 'bx', label='original')
        else:
            plt.plot(soft_scores, precisions, 'ro', label='filtered')
        if not is_secondary:
            plt.hold(True)
            plt.xlabel('No. of valid cards')
            plt.ylabel('Strict Precision (Info)')
            plt.ylim(0, 1)

    plt.close()
    sub_plot(org_weight_score_map)
    if is_filtered:
        sub_plot(filtered_weight_score_map, is_secondary=True)
        plt.legend()
    plt.title('Information Strict Precision Curve')
    plt.savefig(file_path+'.pdf', dpi=300, format='pdf')
    plt.close()


def filter_results_list(results_list, invalid_card_paths):
    filtered_results_list = []
    for curr_result in results_list:
        if curr_result[col_card] in invalid_card_paths:
            continue
        filtered_results_list.append(curr_result)
    return filtered_results_list


def analyze():
    results_list = get_evaluated_results_data(results_file)
    weight_score_map = get_weight_score_card_map(results_list)
    # gen_precision_throughput_curve(weight_score_map, './mitre_june_2015_eval_results/precision_throughput_curve.pdf')
    # weight_score_map = None
    #
    identifiers_list = get_identifiers_list()
    cards_paths_not_related_model_by_grnd = get_num_cards_not_related_to_model_by_grounding(results_list, identifiers_list)
    identifiers_list = None
    filtered_results_list = filter_results_list(results_list, cards_paths_not_related_model_by_grnd)
    results_list = None
    filtered_weight_score_map = get_weight_score_card_map(filtered_results_list)
    #
    gen_precision_throughput_curve(weight_score_map, filtered_weight_score_map, './mitre_june_2015_eval_results/strict_precision_throughput_curve', is_strict=True)
    gen_precision_throughput_curve(weight_score_map, filtered_weight_score_map, './mitre_june_2015_eval_results/generous_precision_throughput_curve', is_strict=False)
    gen_precision_relation_to_model_throughput_curve(weight_score_map, filtered_weight_score_map, './mitre_june_2015_eval_results/relation_to_model_precision_throughput_curve')
    gen_precision_info_throughput_curve(weight_score_map, filtered_weight_score_map, './mitre_june_2015_eval_results/info_precision_throughput_curve')
    filtered_weight_score_map = None


if __name__ == '__main__':
    analyze()
