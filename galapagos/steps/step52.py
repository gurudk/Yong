import cupy as cp
import numpy as np

x = cp.arange(6).reshape(2, 3)
print(x)

y = x.sum(axis=1)
print(y)

n = np.array([1, 2, 3])
c = cp.asarray(n)

assert type(c) == cp.ndarray

c = cp.array([1, 2, 3])
n = cp.asnumpy(n)

assert type(n) == np.ndarray

x = np.array([1, 2, 3])
xp = cp.get_array_module(x)
assert xp == np

x = cp.array([1, 2, 3])
xp = cp.get_array_module(x)
assert xp == cp
