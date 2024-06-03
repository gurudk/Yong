import torch

x = torch.ones((100, 1))
y = torch.randn((1, 10))
z = torch.zeros((100, 20))
print(x)
print(y)
z[:, 0::2] = torch.sin(x * y)
print(z)

position = torch.arange(0, 100).unsqueeze(1)
print(position)
