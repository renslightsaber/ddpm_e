# This file is python codes that I wrote.

import gc
import os
import math
import random
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms

from PIL import Image
import imageio

from dataset import *
from modules import *
from ddpm import *
from trainer import *



# ========================== generate ============================== #

def stack_samples(gen_samples, stack_dim):
    gen_samples = list(torch.split(gen_samples, 1, dim=1))
    for i in range(len(gen_samples)):
        gen_samples[i] = gen_samples[i].squeeze(1)
    return torch.cat(gen_samples, dim=stack_dim)


def gen_img_gif(model,
                n_hold_final = 10,
                t_range = 1000,
                in_size = 32,  
                img_depth = 3, # 3 or 1
                h_num = 2, 
                w_num = 2,
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                save_output_path = "/home/heiscold/ddpm_e/outputs/fashion/",
                save_file_name = '1st_try'
                ):
        
    # Make GIF
    gif_shape = [h_num, w_num] # [2, 2]
    sample_batch_size = gif_shape[0] * gif_shape[1]
    # n_hold_final = 10
    # t_range =1000
    
    # in_size = 32
    # img_depth = 3 or 1

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((sample_batch_size, 1, 32, 32)).to(device)
    sample_steps = torch.arange(t_range-1, 0, -1).to(device)
    for t in sample_steps:
        x = denoise_sample(model, x, t, device)
        if t % 50 == 0:
            gen_samples.append(x)
            
    for _ in range(n_hold_final):
        gen_samples.append(x)
    
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
    
    # Process samples and save as gif
    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], in_size, in_size, img_depth)
    
    # function `stack_samples`
    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)
    
    
    if img_depth == 1:
        # MNIST, FashionMNIST
        final_gen = gen_samples.squeeze(-1).detach().cpu().numpy()
    else:
        # CIFAR10
        final_gen = gen_samples.detach().cpu().numpy()
    
    # Save GIF
    imageio.mimsave(
        save_output_path + f"{save_file_name}.gif",
        list(final_gen),
        fps=5,
    )
    print(f"Shape: [{h_num}x{w_num}]")
    print("GIF SAVED")
    
    # Save IMG
    plt.imshow(final_gen[-1], cmap = 'gray')
    img = Image.fromarray(final_gen[-1])
    img.save(save_output_path + f"{save_file_name}.jpeg")
    print("IMG SAVED")
    
    
# ========================== config = define() ============================== #

def define():
    p = argparse.ArgumentParser()

    # Random Seed
    p.add_argument('--seed', type = int, default = 2024, help="Seed")
    
    # Dataset
    p.add_argument('--dataset_name', type = str, default = "Fashion", help="dataset name")
    
    # File Name when Saved
    p.add_argument('--save_file_name', type = str, default = "1st_try", help="save file name")
    
    # h_num of gif
    p.add_argument('--h_num', type = int, default = 2, help="h_num of gif")
    
    # w_num of gif
    p.add_argument('--w_num', type = int, default = 2, help="w_num of gif")
    
    # Saved Best Model's Epoch
    p.add_argument('--saved_epoch', type = int, default = 365, help="Saved Best Model's Epoch")
    
    # CUDA
    p.add_argument('--device', type = str, default = "cuda", help="CUDA or MPS or CPU?")

    config = p.parse_args()
    return config

# Infer CLI Code (Example: FashionMNIST)
# CUDA_VISIBLE_DEVICES=2 python inference.py --save_file_name 'fashion_t_01' --seed 2024 --h_num 2 --w_num 2 --saved_epoch 504
# CUDA_VISIBLE_DEVICES=2,3 python inference.py --save_file_name 'fashion_t_01' --seed 2024 --h_num 2 --w_num 2 --saved_epoch 504
# /home/heiscold/ddpm_e/ckpts/fashion/ddpm_ep_504.bin

# ========================== main() ============================== #

def main(config):
    
    dataset_name = config.dataset_name # "MNIST", "Fashion", "CIFAR10"
    save_path = f"/home/heiscold/ddpm_e/ckpts/{dataset_name.lower()}/"
    save_output_path = f"/home/heiscold/ddpm_e/outputs/{dataset_name.lower()}/"
    save_file_name = save_file_name= config.save_file_name # 1st try
     
    t_range = 1000
    in_size = 32
    
    # Dataset Name -> img_depth
    if dataset_name == "MNIST" or dataset_name == "Fashion":
        img_depth= 1
    else:
        img_depth= 3
        
    # Random Seed
    seed_num = random.randrange(1, config.seed + 100)
    set_seed(seed= seed_num) 
    
    # Device
    if config.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Model 
    model = DiffusionModel(in_size = in_size,     # 32 
                           t_range = t_range,     # 1000, 
                           img_depth = img_depth, # 1
                           ).to(device)
    
    # Load saved Model
    model.load_state_dict(torch.load(save_path + f'ddpm_ep_{config.saved_epoch}.bin'))
    # model = model.to(device)

    # Generate & Save
    gen_img_gif(model = model.to(device),
                n_hold_final = 10,
                t_range = t_range, # 1000
                in_size = in_size, # 32
                img_depth = img_depth, # 3 or 1
                h_num = config.h_num, # 2
                w_num = config.w_num, # 2
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                save_output_path = save_output_path,
                save_file_name = save_file_name, # 1st try
                )
    
    torch.cuda.empty_cache()
    _ = gc.collect()
    
    print("Generated All.")
    

if __name__ == '__main__':
    config = define()
    main(config)
