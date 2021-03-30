import torch
import torch.nn as nn
import torch.functional as F 
from torch.autograd import Variable
import numpy as np
from utils import to_device

"""
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
"""

class Discriminator(nn.Module):

    def __init__(self, input_size=9):
        super(Discriminator, self).__init__()
        
        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )
    

    def forward(self, x):
        return self.model(x)


    def loss(self, output_real, output_synth):
        return -torch.mean(output_real) + torch.mean(output_synth)



class Generator(nn.Module):

    def __init__(self, latent_dim, input_size=9):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size

        def block(input_size, output_size, normalize=True):
            layers = [nn.Linear(input_size, output_size)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_size, eps=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),  # Unpacking array
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.input_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

    def loss(self, output_synth):
        return -torch.mean(output_synth)



class Wgan(nn.Module):
    
    def __init__(self, args):
        super(Wgan, self).__init__()

        # Miscellaneous
        self.device = args.device

        # Data
        self.batch_size = args.batch_size

        # Models
        self.input_size = args.input_size
        self.latent_dim = args.latent_dim

        self.G = Generator(self.latent_dim, self.input_size).to(self.device)
        self.D = Discriminator(self.input_size).to(self.device)

        # Optimization
        self.n_critic = args.n_critic
        self.clip_value = args.clip_value
        self.lr = args.lr
        self.epochs = args.epochs
        self.optimizer_G = torch.optim.RMSprop(self.G.parameters(), self.lr)
        self.optimizer_D = torch.optim.RMSprop(self.D.parameters(), self.lr)


    def train(self, dataloader):

        # model.train()

        Tensor = torch.cuda.FloatTensor if self.device == "cuda" else torch.FloatTensor

        losses_D = []
        losses_G = []
        for epoch in range(self.epochs):
            for idx, batch in enumerate(dataloader):

                batch = Variable(to_device(batch, self.device))  # Variable ? 
                
                # Train Discriminator 
                self.optimizer_D.zero_grad()
                # Sample noice
                z = Variable(Tensor(np.random.normal(0, 1, size=(batch.shape[0], self.latent_dim))))#, device=self.device))  # Variable ? 
                # Batch of synthetic examples
                synth_batch = self.G(z).detach()  # Why detach ?  
                # Model outputs 
                output_real = self.D(batch)
                output_synth = self.D(synth_batch)
                # Loss 
                loss_D = self.D.loss(output_real, output_synth)
                losses_D.append(loss_D.item())
                loss_D.backward()
                self.optimizer_D.step()
                # Clip weights
                for p in self.D.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

                # Train generator every n_critic iterations
                if idx % self.n_critic == 0:
                    self.optimizer_G.zero_grad()
                    # loss
                    loss_G = self.G.loss(self.D(synth_batch))
                    losses_G.append(loss_G.item())
                    loss_G.backward()
                    self.optimizer_G.step()

                if idx % 100 == 0:
                    print(f"Epoch {epoch}, Iteration {idx}, D loss: {loss_D.item()}, G loss: {loss_G.item()}")
                    # print(f"real example: {batch[0]}")
                    # print(f"sample example: {synth_batch[0]}")
          
        return losses_D, losses_G



class Encoder(nn.Module):

    def __init__(self, input_size=9, embedding_dim=128, hidden=128):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden = hidden

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.embedding_dim),
            nn.Tanh(),
        )
    

    def forward(self, x):
        return self.model(x)



class Decoder(nn.Module):

    def __init__(self, input_size=9, embedding_dim=128, hidden=128):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden = hidden

        self.model = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.input_size),
            nn.ReLU(),
        )
    

    def forward(self, x):
        return self.model(x)



class Autoencoder(nn.Module):

    def __init__(self, input_size=9, embedding_dim=128, hidden=128, epochs=10, batch_size=16, device="cpu"):
        super(Autoencoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size

        self.encoder = Encoder(self.input_size, self.embedding_dim, self.hidden)
        self.decoder = Decoder(self.input_size ,self.embedding_dim, self.hidden)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()

        self.device = device


    def train(self, train_loader, val_loader):

        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):

            train_loss = 0
            val_loss = 0

            # Train
            for idx, batch in enumerate(train_loader):

                target = Variable(to_device(batch, self.device))
                reconstruction = self.decoder(self.encoder(target))

                loss = self.loss(reconstruction, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Val
            for idx, batch in enumerate(val_loader):

                target = Variable(to_device(batch, self.device))
                reconstruction = self.decoder(self.encoder(target))           
                
                loss = self.loss(reconstruction, target)
                val_loss += loss.item()

            train_losses.append(train_loss/len(train_loader))
            val_losses.append(val_loss/len(val_loader))
        
        return train_losses, val_losses
