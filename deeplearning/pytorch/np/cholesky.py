import numpy as np

A = np.array([[9, -6, 3, 24], [-6, 20, -22, -4], [3, -22, 30, -9],
              [24, -4, -9, 99]])

print(A)

L = np.linalg.cholesky(A)

print(L)
