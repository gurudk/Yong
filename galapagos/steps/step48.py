import galapagos.core.datasets as D
import math
import numpy as np
from galapagos.core.models import MLP

import galapagos.core.functions as F
import galapagos.core.optimizers as O

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 0.01

x, t = D.get_spiral(train=True)
print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])

model = MLP((hidden_size, 3))
optimizer = O.SGD(lr).setup(model)

data_size = len(x)

max_iter = math.ceil(data_size / batch_size)
for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))
