import torch

x = torch.round(torch.randn(3, 5), decimals=3)
print(x)

print(str(x))
print(x.tolist())

loss = torch.tensor(-741.63464)
print(loss)
ploss = loss.cpu().detach().item()
print(ploss)
print(loss.tolist())
print(loss.item())

print("............................................")

t1 = torch.tensor([[0.5805, 0.5486, 0.7039, 0.7028],
                   [0.7672, 0.5222, 0.8570, 0.6556],
                   [0.2344, 0.4417, 0.3680, 0.6361],
                   [0.6664, 0.3125, 0.7828, 0.5028]])

for st in t1:
    print(st)

print(t1[:, :2])
print(t1[:, 2:])

btensor = t1[:, 2:] > t1[:, :2]
print(btensor.all())
