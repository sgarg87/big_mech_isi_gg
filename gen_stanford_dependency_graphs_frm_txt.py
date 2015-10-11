import stanford_dependencies as sd


if __name__ == '__main__':
    import sys
    sdg_file_path = sys.argv[1]
    json_file_path = sys.argv[2]
    dot_files_dir_path = sys.argv[3]
    sd_obj = sd.StanfordDependencies()
    sd_obj.load_stanford_dependency_graphs_frm_text(sdg_file_path, json_file_path, dot_files_dir_path)




