import torch.nn as nn
import torch

m = nn.LogSoftmax(dim=1)
input = torch.randn(4, 3)
output = m(input)

print(output)
