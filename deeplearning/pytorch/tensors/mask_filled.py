import torch

ten = torch.tensor([[0, 1.0, 1.0], [1.0, 0, 1.0]])
xten = torch.tensor([[0, 1.0, 1.0], [0, 0, 1.0]])
print(ten.masked_fill(xten == 0.0, 9.9))

torch.masked_fill()
