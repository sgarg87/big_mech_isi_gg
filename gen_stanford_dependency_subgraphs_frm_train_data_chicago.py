import stanford_dependencies as sd


if __name__ == '__main__':
    import sys
    file_path = sys.argv[1]
    const_global_joint_train_dependencies = 'concept_domain_catalyst_joint_train_data_dependencies_chicago'
    sd_obj = sd.StanfordDependencies()
    sd_obj.load_stanford_dependency_graphs_frm_text_nd_gen_subgraphs(
        file_path,
        const_global_joint_train_dependencies=const_global_joint_train_dependencies
    )

