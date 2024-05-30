import torch
import torch.nn.functional as F

x = torch.randn(1, 10, 4)

raw_weights = torch.bmm(x, x.transpose(1, 2))

weights = F.softmax(raw_weights, dim=2)

y = torch.bmm(weights, x)
print(y)
