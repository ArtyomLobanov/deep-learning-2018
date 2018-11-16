import argparse
import logging
import os
import sys

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from homework.vae.vae import VAE
from homework.vae.vae import loss_function
from homework.vae.trainer import Trainer


def get_config():
    parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_dcgan.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=100)
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def main():
    config = get_config()
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=config.data_root, download=True,
                               transform=transform, train=True)
    test_dataset = datasets.MNIST(root=config.data_root, download=True,
                                   transform=transform, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                             num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size,
                                             num_workers=4, pin_memory=True)

    model = VAE(image_size=28)

    trainer = Trainer(model, train_loader, test_loader, Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                      loss_function, device="cuda")

    trainer.train(200, 60)


if __name__ == '__main__':
    main()
