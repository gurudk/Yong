import numpy as np

from galapagos.core.utils import sum_to

from galapagos.core.variable import Variable

x = np.array([[1, 2, 3], [4, 5, 6]])
y = sum_to(x, (1, 3))

print(y)

y = sum_to(x, (2, 1))
print(y)

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)
y.backward(retain_grad=False)
print(x0.grad)
print(x1.grad)
