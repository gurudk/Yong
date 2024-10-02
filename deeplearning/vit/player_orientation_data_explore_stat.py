import json
import numpy as np
import torch

file = "./explored/loss25_data_explored_file_140pth_angle.txt.20240929153227"
val_file = "./explored/loss2555555_val_data_explored_file_140pth_angle.txt.20240929155538"

with open(file, 'r') as rf:
    jd = json.loads(rf.read())

print(len(jd))

list1 = list(filter(lambda x: x[1] < 0.5, jd))
print(len(list1))

np1 = np.array(list1)

a = np.float32(np1[:, 1])

print(a.mean())
