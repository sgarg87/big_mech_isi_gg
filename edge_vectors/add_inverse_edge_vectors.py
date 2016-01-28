#
# Written by Sahil Garg (sahilgar@usc.edu, sahil@isi.edu, sahilvk87@gmail.com)
#
# Sahil Garg, Aram Galstyan, Ulf Hermjakob, and Daniel Marcu. Extracting biomolecular interactions using semantic parsing of biomedical text. In Proc. of AAAI, 2016.
#
# Copyright ISI-USC 2015
#
import edge_labels_propagation as elp


if __name__ == '__main__':
    import sys
    is_amr = bool(sys.argv[1])
    is_overwrite = bool(sys.argv[2])
    #
    print 'is_amr', is_amr
    print 'is_overwrite', is_overwrite
    #
    iev_obj = elp.InverseEdgeVectors(is_amr=is_amr, is_overwrite=is_overwrite)
    iev_obj.process()

