import json
import numpy as np
import torch
from torchvision.ops import box_area


def box_inter_percent(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter / area2


clazz_num = 5
spans = np.linspace(0, 1, num=clazz_num + 1)
interval = 0.2
data = []
box = []
for i in range(clazz_num):
    for j in range(clazz_num):
        box = [spans[j], spans[i], spans[j + 1], spans[i + 1]]
        data.append(box)

data_t = torch.tensor(data)

normalized_annotation_file = "./annotation/annotation_normalized_20240910160828.txt"
probability_annotation_file = "./annotation/annotation_probability_20240910160828_25.txt"
prob_dict = {}
with open(probability_annotation_file, 'w') as wf:
    with open(normalized_annotation_file, 'r') as f:
        obj = json.loads(f.read())
        for key, value in obj.items():
            target = torch.tensor(value)
            target = target[None, :]

            target_arr = box_inter_percent(data_t, target).squeeze().cpu().detach().numpy()
            prob_dict[key] = np.round(target_arr, 4).tolist()
    wf.write(json.dumps(prob_dict))
