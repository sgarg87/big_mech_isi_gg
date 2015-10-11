

def get_inverse_of_edge_label(edge_label):
    edge_label = edge_label.lower()
    if edge_label.endswith('-of'):
        new_edge_key = edge_label[:-3]
    else:
        new_edge_key = edge_label+'-of'
    return new_edge_key
