import torch
import torch.nn as nn

torch.manual_seed(0)
loss = nn.L1Loss()
input = torch.randn(1, 4, requires_grad=True)
target = torch.randn(1, 4)
output = loss(input, target)
# output.backward()

print(input, target, output)
