if __name__ == '__main__':
    import edge_labels_propagation as elp
    import sys
    is_amr = bool(sys.argv[1])
    is_overwrite = bool(sys.argv[2])
    #
    print 'is_amr', is_amr
    print 'is_overwrite', is_overwrite
    #
    iev_obj = elp.InverseEdgeVectors(is_amr=is_amr, is_overwrite=is_overwrite)
    iev_obj.process()

