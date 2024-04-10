import numpy as np

a = np.zeros((2, 3))
b = np.ones((3,))
print(b)
slices = 1
np.add.at(a, slices, b)
print(a)

from galapagos.core import Variable

import galapagos.core.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.get_item(x, 1)
# print(y)
#
# y.backward()
# print(x.grad)

indices = np.array([0, 0, 1])
y = F.get_item(x, indices)
print(y)
x.cleargrad()
y.backward()
print(x.grad)

y = x[1]
print(y)
y = x[:, 2]
print(y)
