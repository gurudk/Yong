import torch

t1 = torch.randn((256, 256))
t2 = 256

print(t1)
print(t2)
print([t2] + t1)
