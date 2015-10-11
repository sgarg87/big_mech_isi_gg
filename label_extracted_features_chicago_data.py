import semi_automated_extraction_features_chicago_data as saefcd


if __name__ == '__main__':
    import sys
    num_cores = int(sys.argv[1])
    # saefcd.label_subgraphs_nd_dump(num_cores)
    saefcd.filter_data_on_ids()
