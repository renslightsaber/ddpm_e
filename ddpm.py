# Github[model.py]: https://github.com/awjuliani/pytorch-diffusion/blob/master/model.py

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *

# =================================================================================================================== #

def beta(t):
    # q(x1:T|x0)가 β1,⋯,βT에 따라 가우시안 noise를 점진적으로 추가하는 Markov chain.

    t_range = 1000
    beta_small = 1e-4
    beta_large = 0.02

    output = beta_small + (t / t_range) * (beta_large - beta_small)
    return output

# =================================================================================================================== #

def alpha(t):
    # αt:=1−βt
    return 1 - beta(t)

# =================================================================================================================== #

def alpha_bar(t):
    # ¯
    # αt:= t∏s=1 αs ## 아오 중복조합 뜻하는 거임
    return math.prod([alpha(j) for j in range(t)])

# =================================================================================================================== #

class DiffusionModel(nn.Module):
    def __init__(self,
                 in_size = 32, # train_dataset.size*train_dataset.size
                 t_range = 1000, # diffusion_steps
                 img_depth = 3, # train_dataset.depth
                 device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
                 ):
        super().__init__()
        self.device = device
        self.in_size = in_size
        self.t_range = t_range

        self.beta_small = 1e-4
        self.beta_large = 0.02

        bilinear = True
        self.inc = DoubleConv(img_depth, 64)

        self.down1 = Down(in_channels = 64, out_channels = 128)
        self.down2 = Down(128, 256)

        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 //factor)

        self.up1 = Up(in_channels = 512, out_channels = 256 // factor, bilinear= bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)

        self.out_c = OutConv(64, img_depth)

        self.sa1 = SAWrapper(h_size=256, num_s =8)
        self.sa2 = SAWrapper(h_size=256, num_s =4)
        self.sa3 = SAWrapper(h_size=128, num_s =8)


    def pos_encoding(self, t, channels, embed_size):
        # bs = 128

        # inv_freq: [64] (64 = channels // 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device = self.device).float() / channels))
        # > tensor([1.0000, 0.8660, 0.7499, 0.6494, 0.5623, ...

        # t.repeat(1, channels // 2) >> Shape: [bs, channels // 2]
        # t.repeat(1, channels // 2) * inv_freq >> Shape: [bs, channels // 2]
        pos_enc_sin = torch.sin( t.repeat(1, channels // 2) * inv_freq)
        pos_enc_cos = torch.cos( t.repeat(1, channels // 2) * inv_freq)

        # pos_enc_sin >> Shape: [bs, channels // 2]
        # pos_enc_cos >> Shape: [bs, channels // 2]

        pos_enc = torch.cat([pos_enc_sin, pos_enc_cos], dim = 1)
        # pos_enc >> Shape: [bs, channels]

        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)
        # pos_enc >> Shape: [bs, channels] -> [bs, channels, 1, 1] -> [128, 128, embed_size, embed_size]

    def forward(self, x, t):
        # Model is U-Net with added positional encoding and self-attention layers.

        # bs = 128
        # t_range = 1000
        # ts = torch.randint(0, t_range, [128], device=device)
        # ts.shape: [128]

        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, channels=128, embed_size= 16)
        # self.down1(x1): [128, 128, 16, 16]
        # self.pos_encoding(t, channels=128, embed_size= 16): [128, 128, 16, 16]

        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        # self.down2(x2): torch.Size([128, 256, 8, 8])
        # self.pos_encoding(t, 256, 8): [128, 256, 8, 8]

        x3 = self.sa1(x3)
        # x3: [128, 256, 8, 8] -> [128, 256, 8, 8]

        # bilinear = True
        # factor = 2 if bilinear else 1
        # down3 = Down(256, 512 // factor)

        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        # self.down3(x3): [128, 256, 4, 4]
        # self.pos_encoding(t, 256, 4): [128, 256, 4, 4]

        x4 = self.sa2(x4)
        # x4: [128, 256, 4, 4]

        # ---------------------------------------------------------- #

        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
        # self.up1(x4, x3): [128, 128, 8, 8]
        # self.pos_encoding(t, 128, 8): [128, 128, 8, 8]
        # x: [128, 128, 8, 8]

        x = self.sa3(x)
        # x: [128, 128, 8, 8]

        x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
        # self.up2(x, x2): [128, 64, 16, 16]
        # self.pos_encoding(t, 64, 16): [128, 64, 16, 16]
        # x: [128, 64, 16, 16]

        x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)
        # self.up3(x, x1): [128, 64, 32, 32]
        # self.pos_encoding(t, 64, 32): [128, 64, 32, 32]
        # x: [128, 64, 32, 32]

        output = self.out_c(x)
        # output: [128, 3, 32, 32]

        return output


# =================================================================================================================== #

@torch.inference_mode()
def denoise_sample(model, x, t, device):

    """
    Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
    """
    with torch.no_grad():
        if t > 1:
            z = torch.randn(x.shape).to(device)
        else:
            z = 0
        e_hat = model(x, t.view(1, 1).repeat(x.shape[0], 1))
        pre_scale = 1 / math.sqrt(alpha(t))
        e_scale = (1 - alpha(t)) / math.sqrt(1 - alpha_bar(t))
        post_sigma = math.sqrt(beta(t)) * z
        x = pre_scale * (x - e_scale * e_hat) + post_sigma
        return x
