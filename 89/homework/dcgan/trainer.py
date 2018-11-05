import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from homework import metric


class DCGANTrainer:

    def __init__(self, discriminator, generator, optimizer_d, optimizer_g, latent_size=100,
                 device='cuda', metrics_dir='metrics', save_root='ckpt', log_dir=None):
        self.net_g = generator
        self.net_d = discriminator
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g
        self.latent_size = latent_size
        self.device = device

        self.metric_dir = metrics_dir
        self.save_root = save_root

        self.net_g.to(device)
        self.net_d.to(device)
        self.net_g.apply(self._weights_init)
        self.net_d.apply(self._weights_init)

        self.writer = SummaryWriter(log_dir=log_dir)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def save(self, epoch):
        os.makedirs(self.save_root, exist_ok=True)
        torch.save(self.net_g.state_dict(), os.path.join(self.save_root, f'generator_epoch_{epoch}.pt'))
        torch.save(self.net_d.state_dict(), os.path.join(self.save_root, f'discriminator_epoch_{epoch}.pt'))

    def train(self, dataloader, n_epoch=25, n_show_samples=8, show_img_every=10, log_metrics_every=100,
              metrics_dataset='cifar10', metrics_to_log=('inception-score', 'mode-score', 'fid')):
        criterion = nn.BCELoss()

        global_step = 0
        for epoch in range(n_epoch):
            for i, data in enumerate(dataloader):

                self.net_d.zero_grad()
                real, _ = data
                real = real.to(self.device)

                target = torch.ones(real.size()[0], device=self.device)

                output = self.net_d(real)
                err_d_real = criterion(output, target)

                noise = torch.randn(real.size()[0], self.latent_size, 1, 1, device=self.device)
                fake = self.net_g(noise)

                if global_step % show_img_every == 0:
                    x = vutils.make_grid(fake[:n_show_samples, :, :, :], normalize=True, scale_each=True)
                    self.writer.add_image('img/fake', x, global_step)

                    y = vutils.make_grid(real[:n_show_samples, :, :, :], normalize=True, scale_each=True)
                    self.writer.add_image('img/real', y, global_step)

                target = torch.zeros(real.size()[0], device=self.device)
                output = self.net_d(fake.detach())
                err_d_fake = criterion(output, target)

                err_d = err_d_real + err_d_fake
                err_d.backward()
                self.optimizer_d.step()

                self.net_g.zero_grad()
                target = torch.ones(real.size()[0], device=self.device)
                output = self.net_d(fake)
                err_g = criterion(output, target)
                err_g.backward()
                self.optimizer_g.step()

                logging.info(f'epoch: [{epoch}/{n_epoch}] iter: [{i}/{len(dataloader)}] loss_D: {err_d:.4f} '
                             f'loss_G: {err_g:.4f}')
                self.writer.add_scalar('data/loss_discriminator', err_d, global_step)
                self.writer.add_scalar('data/loss_generator', err_g, global_step)

                self.net_g.eval()
                if global_step % log_metrics_every == 0:
                    image_size = real.shape[-1]
                    report_dict = metric.compute_metrics(metrics_dataset,
                                                         image_size=image_size,
                                                         metrics_root=Path(self.metric_dir),
                                                         batch_size=dataloader.batch_size, netG=self.net_g)

                    for mtrc in metrics_to_log:
                        self.writer.add_scalar(f'data/{mtrc}', report_dict[mtrc], global_step)
                self.net_g.train()
                global_step += 1

            self.save(epoch)
