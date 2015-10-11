import json
import numpy as np


path = './AImed/aimed_bioc_sentence_relations.json'
new_path = './AImed/aimed_ids.json'

with open(path, 'r') as f:
    json_map = json.load(f)
#
ids = json_map.keys()
ids.sort()
n = len(ids)
print n
ids = np.array(ids)
print ids
ids.reshape(n, 1)
print ids
ids = ids.tolist()
new_ids = []
for curr_id in ids:
    new_ids.append(curr_id+'.')
ids = new_ids
with open(new_path, 'w') as f:
    json.dump(ids, f, indent=4)
