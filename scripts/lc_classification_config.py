

"""
# Code for generating uniform class intervals.
# While test generation, the average of the bucket is used prediction length which is passed to the decoder

import pandas as pd

from collections import Counter, defaultdict

cc = pd.qcut([tokenizer.tokenize(x["span_target"]).__len__() for x in val_set], 6)

cc

span_lengths = [tokenizer.tokenize(x["span_target"]).__len__() for x in val_set]

length_dict = defaultdict(list)

for c, span_length in zip(cc, span_lengths):
    length_dict[c].append(span_length)

cc

for key in sorted(length_dict.keys()):
    print(key)
    print(np.mean(length_dict[key]) )
"""

print("FOLLOWING CONFIGURATIONS ARE FOR DOMINANT ONLY")
length_label_map = {0: 20, 1:28., 2: 33., 3:41., 4:51., 5:81.}
length_label_map_indexed = [20., 28., 33., 41., 51., 81.]
device = "cuda"

def get_length_from_label(length_label):
    global length_label_map
    return length_label_map[length_label]

def get_categories(length):
    if length <= 24.:
        return 0
    elif 24 < length  <= 30.:
        return 1
    elif 30. < length  <= 36.:
        return 2
    elif 36. < length  <= 45.:
        return 3
    elif 45. < length  <= 58.:
        return 4
    
    return 5
