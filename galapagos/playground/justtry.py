import numpy_try as np

from galapagos.core.variable import Variable
from galapagos.core.utils import plot_dot_graph


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
            30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))

z = goldstein(x, y)
z.backward()

x.name = 'x'
y.name = 'y'
z.name = 'z'

print(z.data)
print(x.grad)
print(y.grad)
plot_dot_graph(z, verbose=False, to_file='goldstein.png')
