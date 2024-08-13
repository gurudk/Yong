import torch
import torch.nn as nn

m = nn.Sigmoid()
input = torch.randn(3)
output = m(input)
print(input, output)
