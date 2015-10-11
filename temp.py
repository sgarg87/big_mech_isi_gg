if __name__ == '__main__':
    import csv
    import json
    #
    label = 'label'
    amr_set = 'amr_set'
    count_of_graphs = 'count of graphs'
    count_graphs_map = {}
    count_interactions_map = {}
    with open('./interaction_num_graphs_stats.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            curr_key = (str(row[amr_set]), int(row[label]))
            if curr_key not in count_graphs_map:
                count_graphs_map[curr_key] = 0
                assert curr_key not in count_interactions_map
                count_interactions_map[curr_key] = 0
            count_graphs_map[curr_key] += int(row[count_of_graphs])
            count_interactions_map[curr_key] += 1
    #
    # positive labels statistics
    print '(amr_set, ratio)'
    amr_set_ratio_map = {}
    for curr_key in count_graphs_map:
        assert curr_key in count_interactions_map
        if curr_key[1] == 1:
            curr_ratio = count_graphs_map[curr_key]/float(count_interactions_map[curr_key])
            assert curr_ratio >= 1
            amr_set_ratio_map[curr_key[0]] = curr_ratio
            print '({},{})'.format(curr_key[0], curr_ratio)
    #
    #
    import amr_sets
    ratios_list = []
    for curr_auto_amr_set in amr_sets.amr_sets_list_auto:
        ratios_list.append(amr_set_ratio_map[curr_auto_amr_set])
    with open('./amr_sets_positive_label_num_graphs_ratio.json', 'w') as f:
        json.dump(ratios_list, f)
    print ratios_list


