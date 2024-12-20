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


# First, we define Lightning module for the CVAE model
class CVAEModel(pl.LightningModule):
    # In the constructor we just use the previously defined CVAE model and number of classes
    def __init__(self, latent_dims, n_classes):
        super().__init__()
        self.cvae = CondVariationalAutoencoder(latent_dims, n_classes)
        self.n_classes = n_classes

    # Lightning requires a training step function in which the forward step is executed and loss calculated
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_oh = torch.nn.functional.one_hot(y, num_classes=self.n_classes)

        x_hat = self.cvae(x, y_oh)
        loss = loss = ((x - x_hat) ** 2).sum() + self.cvae.encoder.kl

        # For to see the loss during training, we add the current loss to the logger
        self.log('Training loss', loss, on_step=False, on_epoch=True, logger=False, prog_bar=True)

        return loss

    # Defining the optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Training of the model
latent_dims = 2
model = CVAEModel(latent_dims=latent_dims, n_classes=10)

trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=20)
trainer.fit(model, data)


def plot_reconstructed(autoencoder, r0=(-3, 3), r1=(-3, 3),
                       n=8, number=2, device='cuda'):
    # Define plot array:
    fig, axs = plt.subplots(n, n)

    # Loop over a grid in the latent space
    for i, a in enumerate(np.linspace(*r1, n)):
        for j, b in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[a, b]]).to(device)
            # One-hot encoding of the integer
            y = nn.functional.one_hot(torch.tensor([number]),
                                      num_classes=10).to(device)
            # Forwarding the data through the decoder
            x_hat = autoencoder.decoder(z, y)

            x_hat = x_hat.reshape(28, 28).detach().cpu().numpy()
            axs[i, j].imshow(x_hat)
            axs[i, j].axis('off')
    plt.show()


model = model.to(device)
plot_reconstructed(model.cvae, number=7, device=device)


def plot_latent_cvae(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device), torch.nn.functional.one_hot(torch.tensor(y), num_classes=10).to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


plot_latent_cvae(model.cvae, data)
plt.show()
