import gen_extractor_features_data as gtd
import csv


def gen_joint_triplet_csv():
    data = gtd.load_pickled_merged_data(is_train=True)
    labels_map = data['joint_labels_map']
    with open('train_joint_triplet_data_labels.csv', 'w') as f:
        field_names = ['amr_path', 'label']
        w = csv.DictWriter(f, fieldnames=field_names)
        w.writeheader()
        for curr_path in labels_map:
            curr_label = labels_map[curr_path]
            w.writerow({'amr_path': curr_path, 'label': curr_label})


def gen_protein_state_csv():
    data = gtd.load_pickled_protein_state_data(is_train=True)
    labels_map = data['joint_labels_map']
    with open('train_protein_state_data_labels.csv', 'w') as f:
        field_names = ['amr_path', 'label']
        w = csv.DictWriter(f, fieldnames=field_names)
        w.writeheader()
        for curr_path in labels_map:
            curr_label = labels_map[curr_path]
            w.writerow({'amr_path': curr_path, 'label': curr_label})


if __name__ == '__main__':
    gen_joint_triplet_csv()
    gen_protein_state_csv()

