import numpy as npt

a = npt.arange(9).reshape((3, 3))
npt.random.permutation(a)
print(a)
