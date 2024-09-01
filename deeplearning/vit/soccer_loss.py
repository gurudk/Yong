import torch
import torch.nn.functional as F
import random
from torchvision.ops.boxes import box_area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def loss_func(src_box, target_box):
    return 1 - generalized_box_iou(src_box, target_box)


def gen_test_bbox():
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0

    while True:
        x0 = round(random.uniform(0, 1), 4)
        y0 = round(random.uniform(0, 1), 4)
        x1 = round(x0 + random.uniform(0.4, 0.45), 4)
        y1 = round(y0 + random.uniform(0.55, 0.70), 4)

        if x1 < 1 and y1 < 1:
            break

    return [x0, y0, x1, y1]


def gen_test_bbox():
    x0 = round(random.uniform(0, 1), 4)
    y0 = round(random.uniform(0, 1), 4)
    x1 = round(random.uniform(0, 1), 4)
    y1 = round(random.uniform(0, 1), 4)

    return [x0, y0, x1, y1]


b1 = gen_test_bbox()
b2 = gen_test_bbox()
tensor_box1 = torch.tensor(b1, requires_grad=True).reshape(-1, 4)
tensor_box2 = torch.tensor(b2, requires_grad=True).reshape(-1, 4)

# giou_loss = loss_func(tensor_box1, tensor_box2)

# print(giou_loss)

loss_giou = 1 - torch.diag(generalized_box_iou(
    tensor_box1, tensor_box2))

print(b1, b2)
print(loss_giou)
