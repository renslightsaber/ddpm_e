# ddpm_e
This repository is [`DDPM` (Denoising Diffusion Probabilistic Models)](https://arxiv.org/abs/2006.11239) 🔥Pytorch Code Implementation mainly based on [`awjuliani/pytorch-diffusion`](https://github.com/awjuliani/pytorch-diffusion). `awjuliani`'s DDPM code is easily understandable and written on ⚡ Lightning, and I re-write his DDPM code into 🔥 Pytorch code. I know that his code is better, and this repository is just for my research and study.    

Additionally, for faster or more efficient training, I added some codes from:    
- 🤗 `accelerate`    
- ✍🏻️ `wandb` [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/DDPM_easy_ver?nw=nwuserwako)    

뭐 이 정도면 되것지...?

## Colab ipynb urls for more explanation:
Theses codes are much simpler, and you could train DDPM on Colab GPU (recommend: `L4`, `A100`)
- **DDPM_pytorch_CIFAR10_full_explanation.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NLtYY-5Pk5OQbqZeqA_SEXJoyezqCRbO?usp=sharing)     
- **DDPM_pytorch_CIFAR10.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15VBHrctoAcbDJ36QtAYVF7Tlzee58DvQ?usp=sharing)           
- **DDPM_pytorch_MNIST.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13dfCL1WuEBVOo5dFFzrNuu900NtKDQdS?usp=sharing)     
 

## Generated 
You can check these samples at [`assets`](https://github.com/renslightsaber/ddpm_e/tree/main/assets).
### MNIST, FashionMNIST (`batch_size`: 128, `n_epochs`: 500, 600)
 <img src="/assets/imageedit_1_4023098460.gif" width="15%"></img>
 <img src="/assets/imageedit_3_3572262928.gif" width="15%"></img>
 <img src="/assets/imageedit_5_8576558897.gif" width="15%"></img>
 <img src="/assets/imageedit_7_9457065804.gif" width="15%"></img>
 
### CIFAR10 (`batch_size`: 128, `n_epochs`: 2040)
 <img src="/assets/imageedit_1_4283001522.gif" width="15%"></img>
 <img src="/assets/imageedit_3_8219252803.gif" width="15%"></img>
 <img src="/assets/imageedit_5_2391450170.gif" width="15%"></img>
 <img src="/assets/imageedit_7_4175239486.gif" width="15%"></img>

## How to train `DDPM`?
### [wandb login in CLI](https://docs.wandb.ai/ref/cli/wandb-login)
```python
wandb login --relogin '<your-wandb-API-token>'                  
``` 

### [Train (`train.py`)](https://github.com/renslightsaber/ddpm_e/blob/main/train.py): 🤗`accelerate` code is applied only when `training`.
```bash
CUDA_VISIBLE_DEVICES=2 accelerate launch train.py --dataset_name 'Fashion' --try_name 't_01' --seed 2024 --n_epochs 2000 --batch_size 128 --lr 2e-4 --print_iter 50                
```
- `dataset_name`: you can choose train on one of these datasets; `CIFAR10`, `MNIST`, `FashionMNIST`
- `pr_name`: Project name at `wandb`
- `try_name`: Run name at `wandb`
- `seed`: Random Seed
- `n_epochs`: number of epochs
- `batch_size`: Batch Size
- `lr`: Learning Rate
- `print_iter`: show the `train_loss`, `valid_loss`, `lowest_loss` per `print_iter` epochs
- `saved_epoch`: If you have a saved model ckpt, you can load that model and train. (Default: 0 which means there's no ckpt.)
- `device`: `CUDA`

## How to generate? 
### [Inference (`inference.py`)](https://github.com/renslightsaber/ddpm_e/blob/main/inference.py) 
```bash
CUDA_VISIBLE_DEVICES=2 python inference.py --dataset_name 'Fashion' --save_file_name 'fashion_t_01' --seed 2024 --h_num 2 --w_num 2 --saved_epoch 504            
``` 
- `dataset_name`: you can choose train on one of these datasets; `CIFAR10`, `MNIST`, `FashionMNIST`
- `save_file_name`: file name of img or gif that you generate.
- `seed`: Random Seed
- `h_num`: number of imgs in height
- `w_num`: number of imgs in width
  > total img_size(gif_size): `h_num` x `w_num`
- `saved_epoch`: (Required) You can load the saved model ckpt and use that model for generating imgs or gif. Just input the number of the saved model's `epoch`.
- `device`: `CUDA`
  
## References
- Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [`awjuliani/pytorch-diffusion`](https://github.com/awjuliani/pytorch-diffusion)
