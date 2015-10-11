import amr_sets
import train_extractor as te


class BagOfWordsFrPapers:
    def __init__(self):
        amr_graphs, labels = te.get_data_joint(is_train=True)
        raise NotImplementedError


