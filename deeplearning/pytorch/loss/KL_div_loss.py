import torch
import torch.nn as nn

input = torch.randn(2, 3, requires_grad=True)
target = torch.randn(2, 3)

kl_loss = nn.KLDivLoss(reduction='batchmean')
output = kl_loss(input, target)
output.backward()

print('input: ', input)
print('target: ', target)
print('output: ', output)
