import torch

torch.manual_seed(0)

a = torch.randn(3, 5)
print(a)

print(torch.argmax(a, dim=1))
