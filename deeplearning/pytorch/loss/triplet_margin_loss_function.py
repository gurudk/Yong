import torch
import torch.nn as nn

anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)

triplet_margin_loss = nn.TripletMarginLoss(margin=1.0, p=2)
output = triplet_margin_loss(anchor, positive, negative)
output.backward()

print("anchor: ", anchor)
print("positive: ", positive)
print("negative: ", negative)
print("output: ", output)
