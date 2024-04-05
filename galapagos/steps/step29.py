import numpy as np

from galapagos.core.core import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(2.0))
y = f(x)

y.backward(create_graph=True)
print(x.grad)

gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)

iters = 10

for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)
    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad
    x.data -= gx.data / gx2.data
