import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F  # noqa: F401


class VAE(nn.Module):
    def __init__(self, image_size=28, enc_hidden=400,
                 latent_size=20, dec_hidden=400):
        super().__init__()
        self.latent_size = latent_size

        self.fc1 = nn.Linear(image_size * image_size, enc_hidden)
        self.fc21 = nn.Linear(enc_hidden, latent_size)
        self.fc22 = nn.Linear(enc_hidden, latent_size)
        self.fc3 = nn.Linear(latent_size, dec_hidden)
        self.fc4 = nn.Linear(dec_hidden, image_size * image_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar

    def embed(self, x):
        with torch.no_grad():
            mu, logvar = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu, logvar)
        return z


def loss_function(recon_x, x, mu, logvar):
    """
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + kld
