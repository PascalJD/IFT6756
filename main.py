import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

from wgan import Wgan
from utils import to_device, train_val_test_split

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split


@dataclass
class Arguments:
  # Data
  data_folder: str = '/content/gdrive/MyDrive/IFT6756/data'
  batch_size: int = 16

  # Model
  input_size: int = 9
  latent_dim: int = 100  # How to find this ? 

  # Optimization 
  n_critic: int = 5  # Suggested in Wgan's paper
  clip_value: float = 0.01  
  lr: float = 0.00005
  epochs: int = 10  # More eventually 

  # Experiment
  # exp_id: str = 'debug'
  # log: bool = True
  # log_dir: str = '/content/gdrive/MyDrive/IFT6756/logs'
  # seed: int = 42

  # Miscellaneous
  device: str = 'cuda'
  # num_workers: int = 2
  # progress_bar: bool = False
  # print_every: int = 10


# Load data
train = pd.read_csv('data/TrainValTest/train.csv', index_col=0)
val = pd.read_csv('data/TrainValTest/train.csv', index_col=0)
test = pd.read_csv('data/TrainValTest/train.csv', index_col=0)

train_tensor = torch.tensor(train.values, dtype=torch.float32)
val_tensor = torch.tensor(val.values, dtype=torch.float32)
test_tensor = torch.tensor(test.values, dtype=torch.float32)

def main(args):
  # Data
  train_dataloader = DataLoader(train_tensor,
                                batch_size=args.batch_size,
                                shuffle=False)
  val_dataloader = DataLoader(val_tensor,
                              batch_size=args.batch_size,
                              shuffle=False)
  test_dataloader = DataLoader(test_tensor,
                               batch_size=args.batch_size,
                               shuffle=False)
  
  # Model
  model = Wgan(args)

  # Training
  losses_D, losses_G = model.train(train_dataloader)

  return losses_D, losses_G

args = Arguments(device="cpu")
losses_G, losses_G = main(args)

