import torch
import numpy as np

x1 = torch.tensor([0.3, 0.2, 0.4, 0.4])
x2 = torch.tensor([1280, 720, 1280, 720])

print((x1 * x2).to(dtype=int))

print(np.int_(np.array([0.2, 0.3]) * np.array([100, 200])))
