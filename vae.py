import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms

import numpy as np
import random
import argparse
import sys

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import ion, show

parser = argparse.ArgumentParser(description='Vae on MNIST')
parser.add_argument('--batch-size', type = int, default = 128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--max_epoch', type = int, default = 10, 
                    help='set max epoch')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


def sample_z(args):
    mu, log_sigma = args
    eps = Variable(torch.randn(mu.size()))
    return mu + torch.exp(log_sigma / 2) * eps

class Encoder(nn.Module):
    def __init__(self, in_size, h_size, z_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_size, h_size)
        self.fc2 = nn.Linear(h_size, z_size)
        self.fc3 = nn.Linear(h_size, z_size)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 28 * 28)))
        mu = self.fc2(x)
        log_sigma = self.fc3(x)

        return mu, log_sigma

class Decoder(nn.Module):
    def __init__(self, in_size, h_size, z_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, h_size)
        self.fc2 = nn.Linear(h_size, in_size)
        self.z_size = z_size
    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, self.z_size)))
        return F.sigmoid(self.fc2(x))

class VAE(nn.Module):
    def __init__(self, in_size, h_size, z_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_size, h_size, z_size)
        self.decoder = Decoder(in_size, h_size, z_size)
    
    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = sample_z([mu, log_sigma])
        return self.decoder(z), mu, log_sigma

recon_func = nn.BCELoss()
recon_func.size_average = False

def vae_loss(y_true, y_pred, log_sigma, mu):
    """ Compute loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = recon_func(y_pred, y_true)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * torch.sum(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma, dim = 1)
    kl = torch.sum(kl, dim = 0)
    return recon, kl

def train():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.ToTensor()),
                        batch_size=args.batch_size, shuffle=True)

    vae = VAE(784, 512, 20)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for i in range(args.max_epoch):
        train_loss = 0
        recon_loss = 0
        kl_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(data)
            pred, mu, log_sigma = vae(data) 
            recon, kl = vae_loss(target, pred, mu, log_sigma)
            loss = recon + kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            recon_loss += recon.data[0]
            kl_loss += kl.data[0, 0]
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {}, ({:.0f}%)\tLoss: {:.6f}'.format(
                i, 100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
        print('=====  epoch {}, totoal loss {}, recon loss {}, kl_loss {}'.format(i, train_loss / len(train_loader.dataset), recon_loss / len(train_loader.dataset), kl_loss / len(train_loader.dataset)))
    return vae

def show_result(vae):
    '''Take one test image and show the output
    '''
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                        transform=transforms.ToTensor()),
                        batch_size=args.batch_size, shuffle=False)
    idx = random.randint(0, len(test_loader.dataset))
    test_im = test_loader.dataset[idx][0]
    in_im = test_im.view(28, 28).numpy()
    pred, mu, log_sigma = vae(Variable(test_im))
    out_im = pred.view(28, 28).data.numpy()

    plot_image = np.concatenate((in_im, out_im), axis=1) 
    plt.imshow(plot_image, cmap='gray', interpolation='nearest');
    plt.show()
    



vae = train()
show_result(vae)


