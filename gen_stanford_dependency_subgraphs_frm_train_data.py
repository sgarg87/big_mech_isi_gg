import stanford_dependencies as sd


if __name__ == '__main__':
    import sys
    file_path = sys.argv[1]
    sd_obj = sd.StanfordDependencies()
    sd_obj.load_stanford_dependency_graphs_frm_text_nd_gen_subgraphs(file_path)

