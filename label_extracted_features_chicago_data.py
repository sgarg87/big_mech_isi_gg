import semi_automated_extraction_features_chicago_data as saefcd


if __name__ == '__main__':
    import sys
    num_cores = int(sys.argv[1])
    data = saefcd.load_pickled_data_joint(num_cores)
    saefcd.preprocess_extracted_interactions(data, is_dump_pickled_data=True)
    # saefcd.label_subgraphs_nd_dump(num_cores, is_dump_pickled_data=True)

