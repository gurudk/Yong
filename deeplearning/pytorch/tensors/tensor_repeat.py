import torch

x = torch.tensor([1, 2, 3])
x = x.repeat(4, 1)

print(x.shape)
print(x)
