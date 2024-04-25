import torch
from torch import nn

m = nn.Conv2d(16, 2, 3, stride=2)
m = nn.Conv2d(16, 2, (3, 5), stride=(2, 1), padding=(4, 2))
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

input = torch.randn(20, 16, 50, 100)
output = m(input)

print(output.shape)

m = nn.BatchNorm2d(3)

input = torch.randn(2, 3, 3, 4)
output = m(input)
print(input.shape)
print(output.shape)

m = nn.Conv2d(64, 128, (1, 1))
input = torch.randn(20, 64, 28, 28)
output = m(input)
print(output.shape)
