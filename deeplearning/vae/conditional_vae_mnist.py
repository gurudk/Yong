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
import matplotlib.pyplot as plt

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


class CondVariationalEncoder(nn.Module):

    # The encoder gets the label as an one-hot encoding
    def __init__(self, latent_dims, n_classes):
        super(CondVariationalEncoder, self).__init__()
        # The dimensions of the one-hot encoding are added concatenated to the input
        self.linear1 = nn.Linear(784 + n_classes, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    # The labels are provided as variable `y`
    def forward(self, x, y):
        x = torch.flatten(x, start_dim=1)
        x = x.view(-1, 1 * 28 * 28)
        # Here the label one-hot encoding is concatenated to the image
        x = F.relu(self.linear1(torch.cat((x, y), dim=1)))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class CondVariationalDecoder(nn.Module):

    # The decoder gets the label as an one-hot encoding
    def __init__(self, latent_dims, n_classes):
        super(CondVariationalDecoder, self).__init__()
        # The dimensions of the one-hot encoding are added concatenated to the input
        self.linear1 = nn.Linear(latent_dims + n_classes, 512)
        self.linear2 = nn.Linear(512, 784)

    # Labels are provided as variable `y`
    def forward(self, z, y):
        # Here the label one-hot encoding is concatenated to the image
        z = F.relu(self.linear1(torch.cat((z, y), dim=1)))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class CondVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, n_classes):
        super(CondVariationalAutoencoder, self).__init__()
        self.encoder = CondVariationalEncoder(latent_dims, n_classes)
        self.decoder = CondVariationalDecoder(latent_dims, n_classes)

    def forward(self, x, y):
        z = self.encoder(x, y)
        return self.decoder(z, y)


def train(autoencoder, data, epochs=20, n_classes=10):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in tqdm(range(epochs)):
        for x, y in data:
            x = x.to(device)  # GPU
            number = torch.nn.functional.one_hot(torch.tensor(y), num_classes=n_classes).to(device)
            opt.zero_grad()
            x_hat = autoencoder(x, number)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
        print(loss)
    return autoencoder


# Model training

# We also use two latent dimensions for a simple visualization of the latent space
# Changin the number of latent dimensions will break the later plotting function
latent_dims = 2
cvae = CondVariationalAutoencoder(latent_dims, n_classes=10).to(device)  # GPU
cvae = train(cvae, data, n_classes=10)


# In this plotting function we create a grid in the latent space and use this as an input for the decoder
# Additionally, the variable `number` we provide the interger we want the model to generate
def plot_reconstructed(autoencoder, r0=(-5, 5), r1=(-5, 5), n=12, number=2, device='cuda'):
    w = 28
    img = np.zeros((n * w, n * w))
    for i, a in enumerate(np.linspace(*r1, n)):
        for j, b in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[a, b]]).to(device)
            # One-hot encoding of the integer
            y = torch.nn.functional.one_hot(torch.tensor([number]), num_classes=10).to(device)
            x_hat = autoencoder.decoder(z, y)

            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])


# Generating a variety of 8's from different positions in the latent space.

# Since the VAE loss pushed the model towards a standard normal distribution in the latent space
# we can just use a grid centered around 0
plot_reconstructed(cvae, r0=(-3, 3), r1=(-3, 3), number=8)


def plot_latent_cvae(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device), torch.nn.functional.one_hot(torch.tensor(y), num_classes=10).to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


plot_latent_cvae(cvae, data)
plt.show()
