import numpy as np

import galapagos.core.functions as F
import galapagos.core.optimizers as O

from galapagos.core import Variable
from galapagos.core.models import MLP

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iters = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = O.MomentumSGD(lr, momentum=0.9)
optimizer.setup(model)

for i in range(max_iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)
