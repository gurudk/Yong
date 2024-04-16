import numpy as np

from galapagos.core.models import MLP
from galapagos.core import as_variable
from galapagos.core import Variable

import galapagos.core.functions as F

model = MLP((10, 3))
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)

print(loss)

print("-----------------------------------")

x = np.array([[1, 2, 3]])
y = F.softmax(x, axis=1)
gx = y.backward()

print(y)
print(gx)
