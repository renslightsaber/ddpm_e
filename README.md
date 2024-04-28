# ddpm_e
This repository is [`DDPM` (Denoising Diffusion Probabilistic Models)](https://arxiv.org/abs/2006.11239) 🔥Pytorch Code Implementation mainly based on [`awjuliani/pytorch-diffusion`](https://github.com/awjuliani/pytorch-diffusion). `awjuliani`'s DDPM code is easily understandable and written on ⚡ Lightning, and I re-write his DDPM code into 🔥 Pytorch code. I know that his code is better, and this repository is just for my research and study.    

Additionally, for faster or more efficient training, I added some codes from:    
- 🤗 `accelerate`    
- ✍🏻️ `wandb` [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/DDPM_easy_ver?nw=nwuserwako)    

뭐 이 정도면 되것지...?

## Colab ipynb urls for more explanation:
Theses codes are much simpler, and you could train DDPM on Colab GPU. I recommend L4 or A100.
- **DDPM_pytorch_CIFAR10_full_explanation.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NLtYY-5Pk5OQbqZeqA_SEXJoyezqCRbO?usp=sharing)     
- **DDPM_pytorch_CIFAR10.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15VBHrctoAcbDJ36QtAYVF7Tlzee58DvQ?usp=sharing)           
- **DDPM_pytorch_MNIST.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13dfCL1WuEBVOo5dFFzrNuu900NtKDQdS?usp=sharing)     
 

## Generated (will be updated soon.)
 <img src="/assets/KakaoTalk_Photo_2024-04-28-16-21-52.gif" width="20%"></img>
 <img src="/assets/KakaoTalk_Photo_2024-04-28-16-22-04.gif" width="20%"></img>
 <img src="/assets/fashion_t_01.gif" width="20%"></img>
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

## Inference 
Generate IMG and GIF.
```bash
CUDA_VISIBLE_DEVICES=2 python inference.py --save_file_name 'fashion_t_01' --seed 2024 --h_num 2 --w_num 2 --saved_epoch 504            
``` 

## References
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [`awjuliani/pytorch-diffusion`](https://github.com/awjuliani/pytorch-diffusion)
