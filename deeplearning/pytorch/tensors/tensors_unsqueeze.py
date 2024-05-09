import torch

# unsqueeze just add a new dim, a new batch of one
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)

# out:
# torch.Size([3, 226, 226])
# torch.Size([1, 3, 226, 226])


c = torch.rand(1, 1, 1, 1, 1)
print(c)
cc = c.squeeze(0)
print(cc.shape)

a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)

a = torch.ones(4, 3, 2)

c = a * torch.rand(3, 1)  # 3rd dim = 1, 2nd dim identical to a
print(c)

a = torch.ones(4, 3, 2)
b = torch.rand(3)  # trying to multiply a * b will give a runtime error
c = b.unsqueeze(1)  # change to a 2-dimensional tensor, adding new dim at the end
print(c.shape)
print(a * c)  # broadcasting works again!

batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)

output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)
