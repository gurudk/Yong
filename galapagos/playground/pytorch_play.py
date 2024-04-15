import torch

import cupy

x = torch.rand(5, 4)
print(x)
print(torch.cuda.is_available())
