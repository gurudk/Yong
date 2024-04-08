from galapagos.core.variable import Variable

from galapagos.core.functions import matmul

import numpy as np

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = matmul(x, W)

y.backward()

print(y)
print(x.grad)
print(W.grad)
