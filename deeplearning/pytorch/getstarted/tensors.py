import torch

t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(t.shape)
ft = torch.flatten(t)
print(ft)
print(ft.shape)
ft1 = torch.flatten(t, start_dim=1)
print(ft1.shape)
