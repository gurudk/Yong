import numpy as np

dropout_ratio = 0.6
x = np.ones(10)

mask = np.random.rand(10) > dropout_ratio

y = x * mask
print(mask)
print(x)
print(y)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

import galapagos.core.functions as F

from galapagos.core.variable import test_mode

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)
