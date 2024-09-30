import numpy as np

a = np.array([[1, 2], [2, 1]])
b = np.array([0, 1])
x = np.linalg.solve(a, b)

print(x)
print(np.allclose(np.dot(a, x), b))
