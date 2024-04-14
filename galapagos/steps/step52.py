import cupy as xp

x = xp.arange(6).reshape(2, 3)
print(x)

y = x.sum(axis=1)
print(y)
