import torch

a = torch.ones(2, 2)
b = a

a[0][1] = 561  # we change a...
print(b)  # ...and b is also altered

a = torch.ones(2, 2)
b = a.clone()

assert b is not a  # different objects in memory...
print(torch.eq(a, b))  # ...but still with the same contents!

a[0][1] = 561  # a changes...
print(b)  # ...but b is still all ones

print(torch.eq(a, b))

a = torch.rand(2, 2, requires_grad=True)  # turn on autograd
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)

if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')

if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device='cuda')
    print(gpu_rand)
else:
    print('Sorry, CPU only.')

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)

# to tensor
y = torch.rand(2, 2)
y = y.to(my_device)

x = torch.rand(2, 2)
y = torch.rand(2, 2, device='gpu')
z = x + y  # exception will be thrown
