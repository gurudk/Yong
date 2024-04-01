import numpy as np

from galapagos.core.core import Variable

a = Variable(np.array(2))
b = Variable(np.array(3))

y = a ** b

y.backward()
print(y.data)
print(a.grad)
print(b.grad)
