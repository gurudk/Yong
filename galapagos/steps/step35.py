import numpy as np

from galapagos.core.variable import Variable
from galapagos.core.utils import plot_dot_graph
import galapagos.core.functions as F

x = Variable(np.array(1.0))

y = F.tanh(x)

x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 3

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')
