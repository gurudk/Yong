import numpy as np
from galapagos.core.variable import Variable
import galapagos.core.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward()
print(x.grad)
print(y)
print(y.T)
