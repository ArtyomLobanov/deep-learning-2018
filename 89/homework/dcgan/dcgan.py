import torch.nn as nn


class DCGenerator(nn.Module):

    def __init__(self, image_size, latent_size=100, description_size=256):
        super(DCGenerator, self).__init__()
        layers = [
            nn.ConvTranspose2d(latent_size, description_size, 4, 1, 0, bias=False),
            nn.BatchNorm2d(description_size),
            nn.LeakyReLU(0.005, True)
        ]
        size = 4
        channels = description_size
        while 2 * size < image_size:
            layers.append(nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(channels // 2))
            layers.append(nn.LeakyReLU(0.005, True))
            channels //= 2
            size *= 2

        layers.append(nn.ConvTranspose2d(channels, 3, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, data):
        return self.main.forward(data)


class DCDiscriminator(nn.Module):

    def __init__(self, image_size, initial_channels=16):
        super(DCDiscriminator, self).__init__()
        layers = [
            nn.Conv2d(3, initial_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        size = image_size // 2
        channels = initial_channels
        while size > 4:
            layers.append(nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(channels * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels *= 2
            size //= 2
        layers.append(nn.Conv2d(channels, 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, data):
        return self.main.forward(data)
