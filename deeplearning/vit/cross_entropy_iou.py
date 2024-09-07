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


a = np.linspace(0, 1, num=5)

print(a)

ta = torch.from_numpy(a)
print(ta[0])

tt = torch.linspace(0, 1, 5)
print(tt)

spans = np.linspace(0, 1, num=5)
interval = 0.25
data = []
box = []
for i in range(4):
    for j in range(4):
        box = [spans[j], spans[i], spans[j + 1], spans[i + 1]]
        data.append(box)

data_t = torch.tensor(data)
print(data_t)
target = torch.tensor([[0.4156, 0.5361, 0.7375, 0.7153]])
print('box_area', box_area(target))
ious = box_inter_percent(data_t, target)
print(ious.squeeze().cpu().detach().numpy())
