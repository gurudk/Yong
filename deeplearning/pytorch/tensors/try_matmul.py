import torch

tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)

tensor3 = torch.matmul(tensor1, tensor2)
print(tensor3.shape)
