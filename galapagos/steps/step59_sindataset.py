import numpy as np
import galapagos.core
from galapagos.core.datasets import SinCurve

import matplotlib.pyplot as plt

train_set = SinCurve(train=True)

print(len(train_set))
print(train_set[0])
print(train_set[1])
print(train_set[2])

xs = [example[0] for example in train_set]

ts = [example[1] for example in train_set]

plt.plot(np.arange(len(xs)), xs, label='xs')
plt.plot(np.arange(len(xs)), ts, label='ts')

plt.show()
