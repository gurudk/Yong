import torch

x = torch.randn(4, 4, 4)
x.size()
torch.Size([4, 4])
y = x.view(64)
y.size()
torch.Size([16])
z = x.view(-1, 2)  # the size -1 is inferred from other dimensions
print(z)
z.size()
torch.Size([2, 8])

a = torch.randn(1, 2, 3, 4)
a.size()
torch.Size([1, 2, 3, 4])
b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
b.size()
torch.Size([1, 3, 2, 4])
c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
c.size()
torch.Size([1, 3, 2, 4])
torch.equal(b, c)
