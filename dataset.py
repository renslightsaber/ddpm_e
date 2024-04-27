# Github[data.py]: https://github.com/awjuliani/pytorch-diffusion/blob/master/data.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, 
                 x, # = train_data.data, 
                 dataset = "MNIST"
                 ):

        if dataset == "MNIST" or dataset == "Fashion":
            pad = transforms.Pad(2)
            x = pad(x)
            x = x.unsqueeze(3) # [60000, 32, 32, 1]

            x_pre = ((x / torch.max(x)) * 2.0) - 1.0

            # self.x = torch.tensor(x_pre, dtype = torch.float).permute(0, 3, 1, 2)
            self.x = x_pre.permute(0, 3, 1, 2)

            self.depth = 1 # self.x.shape[1]
            self.size = 32 # self.x.shape[-1]
            
        elif dataset == "CIFAR10":
            self.x = torch.tensor(((x / np.max(x) * 2.0) - 1.0), dtype = torch.float).permute(0, 3, 1, 2)
            # [50000, 32, 32, 3] -> [50000, 3, 32, 32]

            self.depth = 3 # self.x.shape[1]
            self.size = 32 # self.x.shape[-1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index]
