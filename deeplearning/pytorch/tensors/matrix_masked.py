import torch


def create_128x128_loss_mask():
    r1 = torch.cat((torch.tensor([0]), torch.ones(127)), dim=0)
    r2 = torch.cat((torch.tensor([0, 0]), torch.ones(126)), dim=0)
    r3 = torch.zeros(128)
    r3 = r3.repeat(126, 1)

    m = torch.cat((r1.unsqueeze(dim=0), r2.unsqueeze(dim=0), r3), dim=0)

    masked = m.bool()

    return masked


a = torch.arange(0, 128 * 128)
print(a)

b = torch.reshape(a, (128, -1))
print(b)

print(b[create_128x128_loss_mask()].shape)
print(b[create_128x128_loss_mask()].unsqueeze(dim=0))
