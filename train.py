# This file is python codes that I wrote.

import gc
import os
import math
import argparse
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import wandb

# Accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms

from dataset import *
from modules import *
from ddpm import *
from trainer import *


# ========================== config = define() ============================== #

def define():
    p = argparse.ArgumentParser()

    # wandb init
    p.add_argument('--pr_name', type = str, default = "DDPM_easy_ver", help="Project name")
    p.add_argument('--try_name', type = str, default = "1st_try", help="try_name")
    
    # Dataset
    p.add_argument('--dataset_name', type = str, default = "Fashion", help="dataset name")
    
    # training
    p.add_argument('--seed', type = int, default = 2024, help="Seed")
    p.add_argument('--n_epochs', type = int, default = 600, help="Epochs")
    p.add_argument('--batch_size', type = int, default = 128, help="Batch Size")
    p.add_argument('--lr', type = float, default = 2e-4, help="lr")
    p.add_argument('--print_iter', type = int, default = 50, help="Print per N epochs")
    
    # Saved Best Model's Epoch
    p.add_argument('--saved_epoch', type = int, default = 0, help="Saved Best Model's Epoch | Default: 0 which means there's no ckpt.")
    
    # CUDA
    p.add_argument('--device', type = str, default = "cuda", help="CUDA or MPS or CPU?")

    config = p.parse_args()
    return config

# wandb log-in
# wandb login --relogin '<your-wandb-api-token>'

# Train CLI Code (Example: FashionMNIST)
# CUDA_VISIBLE_DEVICES=2 accelerate launch train.py --try_name 't_01' --seed 2024 --n_epochs 2000 --batch_size 128 --lr 2e-4 --print_iter 50
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch train.py --try_name 't_01' --seed 2024 --n_epochs 2000 --batch_size 128 --lr 2e-4 --print_iter 50


# ========================== main() ============================== #

def main(config):
    
    batch_size = config.batch_size # 128
    dataset_name = config.dataset_name # "MNIST", "Fashion", "CIFAR10"
    save_path = f"/home/heiscold/ddpm_e/ckpts/{dataset_name.lower()}/"
    
    t_range = 1000
    in_size = 32
    
    epochs = config.n_epochs # 500
    start_epoch = 0
    end_epoch = start_epoch + epochs
    print_iter =config.print_iter # 50
    lr = config.lr # 2e-4
    
    # Dataset Name -> img_depth
    if dataset_name == "MNIST" or dataset_name == "Fashion":
        img_depth= 1
    else:
        img_depth= 3
        
    # Random Seed
    set_seed(seed=config.seed) # 2024
    
    # Load Data 
    datasets = {"MNIST": MNIST, "Fashion": FashionMNIST, "CIFAR10": CIFAR10 }
    train_data = datasets[dataset_name]("/home/heiscold/ddpm_e/datas/", 
                                        download=True, 
                                        train=True, 
                                        transform=transforms.ToTensor())
    print("Downloaded")
    
    # Dataset, DataLoader
    if dataset_name == "MNIST" or dataset_name == "Fashion":
        cutoff = 50000 
    elif dataset_name == "CIFAR10":
        cutoff = 40000
        
    train_loader = DataLoader(MyDataset(x = train_data.data[:cutoff], dataset = dataset_name), 
                              batch_size = batch_size, 
                              shuffle = True)
    
    valid_loader = DataLoader(MyDataset(x = train_data.data[cutoff:], dataset = dataset_name), 
                              batch_size = batch_size, 
                              shuffle = False)
    print("DataLoader prepared")
    
    # accelerator
    # This is for find unused parameters for avoiding errors
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
                        kwargs_handlers=[kwargs], # log_with="wandb"
                        )
    
    # Device
    if config.device == 'cuda':
        device = accelerator.device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = accelerator.device
        # device = torch.device("cpu")
    
    # Model 
    model = DiffusionModel(in_size = in_size,     # 32 
                           t_range = t_range,     # 1000, 
                           img_depth = img_depth, # 1
                           ).to(device)
    print("model: defined")
    
    # start_epoch = 0
    # end_epoch = start_epoch + epochs
    
    # Load saved Model
    if config.saved_epoch >= 2:
        model.load_state_dict(torch.load(save_path + f'ddpm_ep_{config.saved_epoch}.bin'))
        start_epoch = config.saved_epoch
        end_epoch = start_epoch + epochs
        print("CKPT Loaded")
        
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    print("optimizer: defined")
    
    # accelerator: to accelerator.device
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)
    print("Accelerate Prepared:")
    
    # wandb init
    run = wandb.init(project = config.pr_name, # "DDPM_easy_ver"
                     config = config, 
                     job_type ='Train',
                     name = config.try_name + f"_{dataset_name}", # "1st_try" + "Fashion"
                     anonymous ='must'
                     )
    
    # Run_Train
    model, result = run_train( model = model,
                               accelerator = accelerator,
                               train_loader = train_loader,
                               valid_loader = valid_loader,
                               optimizer = optimizer,
                               t_range = t_range,
                               epochs = end_epoch, # 500, 2040,
                               start_epoch = start_epoch, # 0
                               device = device,
                               print_iter = print_iter, # 50
                               save_path = save_path,
                               verbose = False,
                               save_last = True
                               )
    
    # Train/Valid Loss History
    make_plot(result, stage = "Loss")
    
    # wandb.finish() 
    run.finish()
    
    torch.cuda.empty_cache()
    _ = gc.collect()

    print("Train Completed")
    

if __name__ == '__main__':
    config = define()
    main(config)
