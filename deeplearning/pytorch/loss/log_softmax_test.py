import torch
import torch.nn as nn
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

critiern = nn.CrossEntropyLoss()

m = nn.Softmax(dim=1)
input = torch.randn(2, 3)
output = m(input)

input2 = torch.randn(2, 3)
output2 = m(input2)

print(input, input2)
print(output, output2)

loss_prob = critiern(output, output2)
loss_cross = critiern(input, output2)
loss_input_input2 = critiern(input, input2)

print(loss_prob, loss_cross, loss_input_input2)

print("=============================================")

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5, requires_grad=True)
target_softmax = target.softmax(dim=1)
target_index = torch.argmax(target, dim=1)

loss_index = loss(input, target_index)
loss_softmax = loss(input, target_softmax)

print('input:', input)
print('target:', target)
print('target_index:', target_index)
print('target_softmax:', target_softmax)
print(loss_index, loss_softmax)
