# Github[data.py]: https://github.com/awjuliani/pytorch-diffusion/blob/master/data.py

import numpy as np
from PIL import Image

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
        
        self.dataset = dataset
        
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
            
            self.x = x
            
            self.depth = 3 # self.x.shape[1]
            self.size = 32 # self.x.shape[-1]
            
            augment_horizontal_flip = True
            min1to1 = True
            
            self.transform = transforms.Compose([
                                    transforms.Resize(self.size), 
                                    transforms.RandomHorizontalFlip() if augment_horizontal_flip else torch.nn.Identity(),
                                    transforms.CenterCrop(self.size),
                                    transforms.ToTensor(),  # turn into torch Tensor of shape CHW, 0 ~ 1
                                    transforms.Lambda(lambda x: ((x * 2) - 1)) if min1to1 else torch.nn.Identity()# -1 ~ 1
                                  ])
            # [50000, 32, 32, 3] -> [50000, 3, 32, 32]


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.dataset == "MNIST" or self.dataset == "Fashion":
            return self.x[index]
        
        elif self.dataset == "CIFAR10":
            image = Image.fromarray(self.x[index])
            return self.transform(image)
