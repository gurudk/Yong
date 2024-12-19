import torch

digits = torch.tensor([0, 1, 2, 3])
onshot_tensor = torch.nn.functional.one_hot(digits, 10)
print(onshot_tensor)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import lightning.pytorch as pl

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt;

plt.rcParams['figure.dpi'] = 200
from utils import plot_latent

device = "cuda"  # Use "cpu" if no GPU is available.
dataset_path = '~/datasets'

# Creating a dataloader
data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(dataset_path,
                               transform=torchvision.transforms.ToTensor(),
                               download=True),
    batch_size=128,
    shuffle=True)


# Code from: https://github.com/avandekleut/avandekleut.github.io/blob/master/_posts/2020-05-14-vae.md
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

        self.kl = 0
        self.device = device

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, device)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_vae(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in tqdm(range(epochs)):
        for x, y in data:
            x = x.to(device)  # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
    return autoencoder


latent_dims = 2
vae = VariationalAutoencoder(latent_dims, device).to(device)  # GPU
vae = train_vae(vae, data)

plot_latent(vae, data)
plt.show()
