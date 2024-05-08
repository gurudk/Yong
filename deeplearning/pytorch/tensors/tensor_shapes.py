import torch

x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

# zeros_like_x = torch.zeros_like(x)
# print(zeros_like_x.shape)
# print(zeros_like_x)
#
# ones_like_x = torch.ones_like(x)
# print(ones_like_x.shape)
# print(ones_like_x)
#
rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)

some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)

a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)

ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)

# a = torch.rand(2, 3)
# b = torch.rand(3, 2)
#
# print(a * b)


rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)

a = torch.ones(4, 3, 2)

b = a * torch.rand(3, 2)  # 3rd & 2nd dims identical to a, dim 1 absent
print(a)
print(b)

c = a * torch.rand(3, 1)  # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(1, 2)  # 3rd dim identical to a, 2nd dim = 1
print(d)

# error examples

# a =     torch.ones(4, 3, 2)
#
# b = a * torch.rand(4, 3)    # dimensions must match last-to-first
#
# c = a * torch.rand(   2, 3) # both 3rd & 2nd dims different
#
# d = a * torch.rand((0, ))   # can't broadcast with an empty tensor
