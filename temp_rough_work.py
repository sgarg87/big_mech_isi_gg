import pickle



if __name__ == '__main__':
    with open('./labeled_chicago_concept_domain_catalyst_data.pickle' 'rb') as f:
        obj = pickle.load(f)
        print len(obj['paths_map'])


