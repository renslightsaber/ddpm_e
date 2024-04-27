# This file is python codes that I wrote.

import gc
import os
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from modules import *
from ddpm import *


# =================================================================================================================== #

def set_seed(seed=2024):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
# =================================================================================================================== #

def train_fn(model, # = model,
             accelerator, # from accelerate 
             data_loader, # = DataLoader(MyDataset(x = train_data.data[:30000]), batch_size = 128, shuffle = True),
             optimizer, # = optimizer,
            #  in_size = 32, # train_dataset.size*train_dataset.size
             t_range = 1000, # diffusion_steps
            #  img_depth = 3, # train_dataset.depth
             epoch = 0,
             verbose = False,
             device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
             ):

    """
    Corresponds to Algorithm 1 from (Ho et al., 2020).
    """
    
    model.train()
    
    if verbose:
        bar = tqdm(data_loader, total = len(data_loader))
    else:
        bar = data_loader

    train_sum_loss = 0
    dataset_size = 0

    for data in bar:
        bs = data.shape[0] # batch_size
        data = data.to(device)

        # We need to get a `loss`
        # First, We need to write a code following equations of Algorithm 1
        ts = torch.randint(0, t_range, [bs], device=device)
        noise_imgs = []
        epsilons = torch.randn(data.shape, device = device)
        for i in range(len(ts)):
            # if i == 10 >> ts[10]: 755
            # i: 0 ~ 128

            a_hat = alpha_bar(ts[i])
            # a_hat: 0.003123266397556902
            noise_imgs.append(
                 (math.sqrt(a_hat) * data[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
                # data[i].shape [ch, h, w]
                # epsilons[i]: [ch, h, w]

                # (math.sqrt(a_hat) * data[i]): [3, 32, 32]
                # (math.sqrt(1 - a_hat) * epsilons[i]): [3, 32, 32]
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)

        ## Model Predicted
        # e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
        e_hat = model(noise_imgs.to(device), ts.unsqueeze(-1).type(torch.float).to(device))
        # e_hat >> Shape: torch.Size([128, 3, 32, 32])

        # Get Loss
        loss = nn.functional.mse_loss(e_hat.reshape(-1, model.in_size), epsilons.reshape(-1, model.in_size))

        # Back-propagation
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        dataset_size += bs
        train_sum_loss += float(loss.item() * bs)
        train_loss = train_sum_loss / dataset_size

        if verbose:
            bar.set_postfix(Epoch = epoch, Train_loss = train_loss, LR = optimizer.param_groups[0]['lr'])

    gc.collect()
    return train_loss


# =================================================================================================================== #

@torch.inference_mode()
def valid_fn(model, # = model,
             data_loader, # = DataLoader(MyDataset(x = train_data.data[30000:]), batch_size = 128, shuffle = True),
            #  in_size = 32, # train_dataset.size*train_dataset.size
             t_range = 1000, # diffusion_steps
            #  img_depth = 3, # train_dataset.depth
             epoch = 0,
             verbose = False,
             device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
             ):

    """
    Corresponds to Algorithm 1 from (Ho et al., 2020).
    """
    
    model.eval()
    
    if verbose:
        bar = tqdm(data_loader, total = len(data_loader))
    else:
        bar = data_loader

    valid_sum_loss = 0
    dataset_size = 0
    with torch.no_grad():
        for data in bar:
            bs = data.shape[0]
            data = data.to(device)

            # We need to get a `loss`
            # First, We need to write a code following equations of Algorithm 1
            ts = torch.randint(0, t_range, [bs], device=device)
            noise_imgs = []
            epsilons = torch.randn(data.shape, device = device)
            for i in range(len(ts)):
                # if i == 10 >> ts[10]: 755
                # i: 0 ~ 128

                a_hat = alpha_bar(ts[i])
                # a_hat: 0.003123266397556902
                noise_imgs.append(
                    (math.sqrt(a_hat) * data[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
                    # data[i].shape [ch, h, w]
                    # epsilons[i]: [ch, h, w]

                    # (math.sqrt(a_hat) * data[i]): [3, 32, 32]
                    # (math.sqrt(1 - a_hat) * epsilons[i]): [3, 32, 32]
                )
            noise_imgs = torch.stack(noise_imgs, dim=0)

            ## Model Predicted
            # e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
            e_hat = model(noise_imgs.to(device), ts.unsqueeze(-1).type(torch.float).to(device))
            # e_hat >> Shape: torch.Size([128, 3, 32, 32])

            # Get Loss
            loss = nn.functional.mse_loss(e_hat.reshape(-1, model.in_size), epsilons.reshape(-1, model.in_size))

            dataset_size += bs
            valid_sum_loss += float(loss.item() * bs)
            valid_loss = valid_sum_loss / dataset_size

            if verbose:
                bar.set_postfix(Epoch = epoch, Valid_loss = valid_loss)

    gc.collect()
    return valid_loss


# =================================================================================================================== #

# Run Train
def run_train(model, # = model,
              accelerator, # from `accelerate`
              train_loader, # =DataLoader(MyDataset(x = train_data.data[:40000]), batch_size = 128, shuffle = True),
              valid_loader, # =DataLoader(MyDataset(x = train_data.data[40000:]), batch_size = 128, shuffle = False),
              optimizer, # = optimizer,
              t_range =1000,
              epochs = 100,
              start_epoch = 0, # Default:0 (Assume there is no ckpt) 
              device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
              print_iter = 10,
              save_path = "./",
              verbose = False,
              save_last = True
            #   early_stop =25
              ):

    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("INFO: GPU - {}\n".format(torch.cuda.get_device_name()))
        
    result = dict()
    lowest_loss, lowest_epoch = np.inf, np.inf
    train_hs, valid_hs = [], []

    for epoch in range(start_epoch, epochs):
        gc.collect()

        train_loss = train_fn(model.to(device), accelerator, train_loader, optimizer, t_range, epoch, verbose, device)
        valid_loss = valid_fn(model.to(device), valid_loader, t_range, epoch, verbose, device)

        # Getting Loss for Visualization
        train_hs.append(train_loss)
        valid_hs.append(valid_loss)

        # Log the metrics
        wandb.log({"train/loss": train_loss})
        wandb.log({"eval/loss": valid_loss})
        
        # monitoring
        if (epoch + 1) % print_iter == 0:
            print(f"Epoch{epoch + 1} | Train Loss:{train_loss:.3e} | Valid Loss:{valid_loss:.3e} | Lowest Loss:{lowest_loss:.3e}|")

        # Lowest Loss 갱신  - Valid Loss 기준
        if valid_loss < lowest_loss:
            lowest_loss = valid_loss
            lowest_epoch = epoch
            
            if epoch >= print_iter * 2:    
                accelerator.wait_for_everyone()
                # Unwrap: model
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model_state_dict = unwrapped_model.state_dict() 
                accelerator.save(unwrapped_model_state_dict, save_path + f'ddpm_ep_{epoch}.bin')
                # torch.save(model.state_dict(), save_path + f'ddpm_ep_{epoch}.bin')
                
        # else:
        #     if early_stop > 0 and lowest_epoch+ early_stop < epoch +1:
        #         print("삽질중")
        #         print("There is no improvement during %d epochs" % early_stop)
        #         break
        
    if save_last:
        last_epoch = epochs - 1
        
        accelerator.wait_for_everyone()
        # Unwrap: model
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model_state_dict = unwrapped_model.state_dict() 
        accelerator.save(unwrapped_model_state_dict, save_path + f'ddpm_ep_{last_epoch}.bin')
        # torch.save(model.state_dict(), save_path + f'ddpm_ep_{last_epoch}.bin')
        print("Save Last: Completed")
        
    print("The Best Validation Loss=%.3e at %d Epoch" % (lowest_loss, lowest_epoch))

    # model.load_state_dict(torch.load('./ddpm.bin'))

    result["Train Loss"] = train_hs
    result["Valid Loss"] = valid_hs

    torch.cuda.empty_cache()
    _ = gc.collect()
    
    return model, result


# =================================================================================================================== #

def make_plot(result, stage = "Loss"):
    ## Train/Valid History
    plot_from = 0

    if stage == "Loss":
        trains = 'Train Loss'
        valids = 'Valid Loss'

    elif stage == "Acc":
        trains = "Train Acc"
        valids = "Valid Acc"

    elif stage == "F1":
        trains = "Train F1"
        valids = "Valid F1"

    plt.figure(figsize=(10, 6))

    plt.title(f"Train/Valid {stage} History", fontsize = 20)

    ## Modified for converting Type
    if type(result[trains][0]) == torch.Tensor:
        result[trains] = [num.detach().cpu().item() for num in result[trains]]
        result[valids] = [num.detach().cpu().item() for num in result[valids]]

    plt.plot(
        range(0, len(result[trains][plot_from:])),
        result[trains][plot_from:],
        label = trains
        )

    plt.plot(
        range(0, len(result[valids][plot_from:])),
        result[valids][plot_from:],
        label = valids
        )

    plt.legend()
    if stage == "Loss":
        plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    
# =================================================================================================================== #

