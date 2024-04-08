import numpy as np

from galapagos.core.variable import Variable
import galapagos.core.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x
x, y = Variable(x), Variable(y)
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def prefict(x):
    y = F.matmul(x, W) + b
    return y


lr = 0.1
iters = 1000

for i in range(iters):
    y_pred = prefict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)
