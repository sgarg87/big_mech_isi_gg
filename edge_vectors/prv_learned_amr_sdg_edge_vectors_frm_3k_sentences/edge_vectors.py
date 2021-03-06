import pickle
from ... import config
from .. import edge_labels as el
from ... import constants_absolute_path as cap


class EdgeVectorsAmrSdg:
    def __init__(self, is_amr_edge2vec, is_dependencies2vec):
        assert is_amr_edge2vec or is_dependencies2vec
        #
        if is_amr_edge2vec:
            self.amr_edge2vec_map = None
        #
        if is_dependencies2vec:
            self.dependencies_edge2vec_map = None
        #
        if is_amr_edge2vec or is_dependencies2vec:
            self.edge2vec_map = {}
        else:
            self.edge2vec_map = None
        #
        # loading word vectors for edges
        #
        # AMR
        if is_amr_edge2vec:
            print 'Loading amr edge vectors ...'
            amr_edge_label_word_vec_pickled_file_path = self.get_edge_label_vectors_file_path(is_amr=True)
            with open(cap.absolute_path+amr_edge_label_word_vec_pickled_file_path, 'r') as f_amr_edge_word2vec:
                amr_edge2vec_map = pickle.load(f_amr_edge_word2vec)
            assert amr_edge2vec_map is not None and amr_edge2vec_map
            self.edge2vec_map.update(amr_edge2vec_map)
        #
        # SDG
        if is_dependencies2vec:
            print 'loading dependencies edge vectors ...'
            dependencies_edge_label_word_vec_pickled_file_path = self.get_edge_label_vectors_file_path(is_amr=False)
            with open(cap.absolute_path+dependencies_edge_label_word_vec_pickled_file_path, 'r') as f_dependencues_edge_word2vec:
                dependencies_edge2vec_map = pickle.load(f_dependencues_edge_word2vec)
            assert dependencies_edge2vec_map is not None and dependencies_edge2vec_map
            self.edge2vec_map.update(dependencies_edge2vec_map)

    def get_edge_label_vectors_file_path(self, is_amr):
        file_path = './edge_labels_vectors_propagated'
        if is_amr:
            file_path += '_fr_amrs'
        else:
            file_path += '_fr_dependencies'
        #
        file_path += '.pickle'
        return file_path

    def get_edge_vector(self, edge_label):
        assert self.edge2vec_map is not None and self.edge2vec_map
        if edge_label is None:
            return None
        edge_label = edge_label.strip()
        edge_label_lower = edge_label.lower()
        edge_label_upper = edge_label.upper()
        #
        edge_label_inverse = el.get_inverse_of_edge_label(edge_label)
        edge_label_inverse_lower = edge_label_inverse.lower()
        edge_label_inverse_upper = edge_label_inverse.upper()
        try:
            if edge_label in self.edge2vec_map:
                return self.edge2vec_map[edge_label]
            elif edge_label_lower in self.edge2vec_map:
                return self.edge2vec_map[edge_label_lower]
            elif edge_label_upper in self.edge2vec_map:
                return self.edge2vec_map[edge_label_upper]
            elif edge_label_inverse in self.edge2vec_map:
                return self.edge2vec_map[edge_label_inverse]
            elif edge_label_inverse_lower in self.edge2vec_map:
                return self.edge2vec_map[edge_label_inverse_lower]
            elif edge_label_inverse_upper in self.edge2vec_map:
                return self.edge2vec_map[edge_label_inverse_upper]
            else:
                if config.debug:
                    print '***********************************************'
                    print 'error getting vector for edge label ', edge_label
                    print edge_label
                    print edge_label_lower
                    print edge_label_upper
                    print edge_label_inverse
                    print edge_label_inverse_lower
                    print edge_label_inverse_upper
                    print '***********************************************'
        except UnicodeDecodeError as ude:
            if config.debug:
                print 'error getting vector for edge label ', edge_label
                print ude.message


