#
# Written by Sahil Garg (sahilgar@usc.edu, sahil@isi.edu, sahilvk87@gmail.com)
#
# Sahil Garg, Aram Galstyan, Ulf Hermjakob, and Daniel Marcu. Extracting biomolecular interactions using semantic parsing of biomedical text. In Proc. of AAAI, 2016.
#
# Copyright ISI-USC 2015
#

def get_inverse_of_edge_label(edge_label):
    edge_label = edge_label.lower()
    if edge_label.endswith('-of'):
        new_edge_key = edge_label[:-3]
    else:
        new_edge_key = edge_label+'-of'
    return new_edge_key
