import json
import numpy as np
import torch

file = "./explored/new_explore_140th.txt.20241003090118"
val_file = "./explored/new_data_loss05_valdata.txt.20241003123727"

with open(val_file, 'r') as rf:
    jd = json.loads(rf.read())

print(len(jd))

list1 = list(filter(lambda x: x[1] < 0.5, jd))
print(len(list1))

np1 = np.array(list1)

a = np.float32(np1[:, 1])

print(a.mean())
