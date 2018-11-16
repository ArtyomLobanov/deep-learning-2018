import logging
import os

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class Trainer:

    def __init__(self, model, train_loader, test_loader, optimizer, loss_function, image_size=28, device='cpu'):
        self.model = model
        model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.image_size=image_size
        self.device = device
        self.writer = SummaryWriter()

    def train(self, n_epoch, log_interval):
        self.model.train()
        global_step = 0
        for epoch in range(n_epoch):
            epoch_loss = 0

            for batch_idx, (data, _) in enumerate(self.train_loader):
                self.model.zero_grad()
                data = data.to(self.device)
                res, mu, logvar = self.model(data)

                train_loss = self.loss_function(res, data, mu, logvar)
                train_loss.backward()

                epoch_loss += train_loss
                norm_train_loss = train_loss / len(data)

                self.optimizer.step()
                if batch_idx % log_interval == 0:
                    msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        norm_train_loss)
                    logging.info(msg)
                    self.writer.add_scalar(tag='data/train_loss',
                                           scalar_value=norm_train_loss,
                                           global_step=global_step)
                    self.plot_generated(global_step, data, res, 'img/images')

                global_step += 1

            epoch_loss /= len(self.train_loader.dataset)
            logging.info(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}')
            self.writer.add_scalar(tag='data/train_epoch_loss',
                                   scalar_value=epoch_loss,
                                   global_step=epoch)

    def test(self, epoch, batch_size, log_interval):
        self.model.eval()
        test_epoch_loss = 0
        step = 0

        for batch_idx, (data, _) in enumerate(self.test_loader):
            data = data.to(self.device)
            res, mu, logvar = self.model(data)

            test_loss = self.loss_function(res, data, mu, logvar)
            test_epoch_loss += test_loss

            if batch_idx % log_interval == 0:
                msg = 'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.test_loader.dataset),
                           100. * batch_idx / len(self.test_loader),
                           test_loss / len(data))
                logging.info(msg)

                batches_per_epoch_test = len(self.test_loader.dataset) // batch_size
                self.writer.add_scalar(tag='data/test_loss',
                                       scalar_value=test_loss / len(data),
                                       global_step=batches_per_epoch_test * (epoch - 1) + batch_idx)
                self.plot_generated(step, data, res, 'img/test_images')
            step += 1

        test_epoch_loss /= len(self.test_loader.dataset)
        logging.info('====> Test set loss: {:.4f}'.format(test_epoch_loss))
        self.writer.add_scalar(tag='data/test_epoch_loss',
                               scalar_value=test_epoch_loss,
                               global_step=epoch)

    def plot_generated(self, step_number, origin, generated, tag, image_count=10):
        examples = torch.cat([origin[:image_count, :, :, :],
                              generated[:image_count].view(image_count, 1, self.image_size, self.image_size)], dim=0)
        image = vutils.make_grid(examples, scale_each=True, nrow=10)
        self.writer.add_image(tag, image, step_number)

    def save(self, checkpoint_path):
        dir_name = os.path.dirname(checkpoint_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
