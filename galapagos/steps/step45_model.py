import numpy as np
from galapagos.core import Variable, Model
from galapagos.core.models import MLP

import galapagos.core.functions as F
import galapagos.core.layers as L

np.random.seed(0)
x = np.random.rand(100, 1)

y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid_simple(self.l1(x))
        y = self.l2(y)
        return y


model = MLP((hidden_size, 1))

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 100 == 0:
        print(loss)
