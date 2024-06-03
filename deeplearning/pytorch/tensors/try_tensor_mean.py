import torch

a = torch.randn(4, 5)
print(a)

print(torch.sum(a, 1))

x = torch.arange(0, 20, dtype=float)
print(x)

print(x.reshape((4, 5)))
print(torch.mean(x.reshape((4, 5)), 0))
