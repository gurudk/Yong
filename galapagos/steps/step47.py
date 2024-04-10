import numpy as np

from galapagos.core.models import MLP
from galapagos.core import as_variable

import galapagos.core.functions as F


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


model = MLP((10, 3))
x = np.array([[0.2, -0.4]])
y = model(x)
yp = softmax1d(y)

print(yp)
