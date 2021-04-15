import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import numpy as np
from utils import to_device

"""
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
"""

# W-MedGan

class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.input_dim = args.input_dim
        self.embedding_dim = args.embedding_dim
        self.hidden = args.hidden_D

        self.input_layer = nn.Linear(self.input_dim, self.hidden[0])
        self.input_activation = nn.ReLU()
        self.layers = nn.ModuleList()
        for i in range(1, len(self.hidden)):
            self.layers.append(nn.Linear(self.hidden[i-1], self.hidden[i]))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(self.hidden[-1], 1)

        self.logs = {"train loss": [], "val loss": []}

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_activation(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def loss(self, y_real, y_synthetic):
        return -torch.mean(y_real) + torch.mean(y_synthetic)


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()

        self.random_dim = args.random_dim
        self.embedding_dim = args.embedding_dim
        self.hidden = args.hidden_G
        self.is_finetuning = args.is_finetuning

        self.decoder = args.decoder.to(args.device)
        if not self.is_finetuning:
            for params in self.decoder.parameters():
                params.require_grad = False

        self.input_layer = nn.Linear(self.random_dim, self.hidden[0])
        self.input_activation = nn.ReLU()
        self.layers = nn.ModuleList()
        for i in range(1, len(self.hidden)):
            self.layers.append(nn.Linear(self.hidden[i-1], self.hidden[i]))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(self.hidden[-1], self.embedding_dim)
        self.output_activation = nn.Tanh()

        self.logs = {"train loss": [], "val loss": []}

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_activation(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        x = self.decoder(x)
        return x

    def loss(self, y_synthetic):
        return -torch.mean(y_synthetic)


class GAN(nn.Module):

    def __init__(self, args):
        super(GAN, self).__init__()

        self.random_dim = args.random_dim
        self.embedding_dim = args.embedding_dim

        self.G = Generator(args)
        self.D = Discriminator(args)

        self.logs = {"approx. EM distance": []}


# Autoencoder

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden = args.hidden

        # Layers
        self.input_layer = nn.Linear(self.input_dim, self.hidden[0])
        self.layers = nn.ModuleList()
        for i in range(1, len(self.hidden)):
            self.layers.append(nn.Linear(self.hidden[i-1], self.hidden[i]))
        self.output_layer = nn.Linear(self.hidden[-1], self.latent_dim)

        # Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        x = self.output_layer(x)
        x = self.tanh(x)
        return x


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden = args.hidden

        # Layers
        self.input_layer = nn.Linear(self.latent_dim, self.hidden[-1])
        self.layers = nn.ModuleList()
        for i in range(len(self.hidden)-1, 0, -1):
            self.layers.append(nn.Linear(self.hidden[i], self.hidden[i-1]))
        self.output_layer = nn.Linear(self.hidden[0], self.input_dim)

        # Activations
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self, args):
        super(Autoencoder, self).__init__()

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden = args.hidden

        self.device = args.device

        self.encoder = Encoder(args).to(self.device)
        self.decoder = Decoder(args).to(self.device)

        self.criterion = nn.MSELoss(reduction="sum")

        self.logs = {"train loss": [], "val loss": []}

    def forward(self, x):
        return self.decoder(self.encoder(x))
