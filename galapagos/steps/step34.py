import numpy as np
import matplotlib.pyplot as plt
import galapagos.core.functions as F
from galapagos.core import Variable

x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
logs = [y.data]
y.backward(create_graph=True)
for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(x.grad)

labels = ["y=sin(x)", "y'", "y''", "y''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])

plt.legend(loc="lower right")
plt.savefig("grad.png")
plt.show()
