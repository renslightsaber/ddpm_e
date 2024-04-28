# ddpm_e
This repository is [`DDPM` (Denoising Diffusion Probabilistic Models)](https://arxiv.org/abs/2006.11239) 🔥Pytorch Code Implementation mainly based on [`awjuliani/pytorch-diffusion`](https://github.com/awjuliani/pytorch-diffusion). `awjuliani`'s DDPM code is easily understandable and written on ⚡ Lightning, and I re-write his DDPM code into 🔥 Pytorch code. I know that his code is better, and this repository is just for my research and study. Additionally I added some codes for faster or more efficient training:    
- 🤗 `accelerate`    
- ✍🏻️ `wandb` [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/DDPM_easy_ver?nw=nwuserwako)    

뭐 이 정도면 되것지

## Generated (will be updated soon.)
 <img src="/assets/fashion_t_01.gif" width="20%"></img>
 <img src="/assets/fashion_t_02.gif" width="20%"></img>
 <img src="/assets/fashion_t_04.gif" width="20%"></img>

## How to train `DDPM`?
### [wandb login in CLI interface](https://docs.wandb.ai/ref/cli/wandb-login)
```python
wandb login --relogin '<your-wandb-API-token>'                  
``` 

### Train
```bash
CUDA_VISIBLE_DEVICES=2 accelerate launch train.py --dataset_name 'Fashion' --try_name 't_01' --seed 2024 --n_epochs 2000 --batch_size 128 --lr 2e-4 --print_iter 50                
``` 
- Datasets: `CIFAR10`, `MNIST`, `FashionMNIST`

## Inference (Generate IMG or GIF)
```bash
CUDA_VISIBLE_DEVICES=2 python inference.py --save_file_name 'fashion_t_01' --seed 2024 --h_num 2 --w_num 2 --saved_epoch 504            
``` 

## References
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [`awjuliani/pytorch-diffusion`](https://github.com/awjuliani/pytorch-diffusion)
