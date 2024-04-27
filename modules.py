# Github[modules.py]: https://github.com/awjuliani/pytorch-diffusion/blob/master/modules.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, h_size):
        super().__init__()
        self.h_size = h_size
        # nn.MultiheadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        self.mha = nn.MultiheadAttention(embed_dim = h_size, num_heads = 4, batch_first = True)
        self.layer_norm = nn.LayerNorm([h_size])
        self.ff_self = nn.Sequential(nn.LayerNorm([h_size]), nn.Linear(h_size, h_size), nn.GELU(), nn.Linear(h_size, h_size))

    def forward(self, x):
        # x: [bs, seq_len, h_dim]

        x_ln = self.layer_norm(x)
        # x: [bs, seq_len, h_dim]

        # attn_output, attn_output_weights
        attn_output, _ = self.mha(x_ln, x_ln, x_ln)
        # attn_output: [bs, seq_len, h_dim]

        # Residual Connection
        attn_output = attn_output + x
        # attn_output: [bs, seq_len, h_dim]
        # x: [bs, seq_len, h_dim]

        # FeedForwardLayer + Residual Connection
        attn_output = self.ff_self(attn_output) + attn_output
        
        return attn_output
    
class SAWrapper(nn.Module):
    def __init__(self, h_size, num_s):
        super().__init__()
        self.sa = nn.Sequential(*[SelfAttention(h_size) for _ in range(1)]) # nn.Sequential(SelfAttention_Layer)
        self.num_s = num_s
        self.h_size = h_size

    def forward(self, x):
        # x: [bs, seq_len, h_dim]

        x = x.view(-1, self.h_size, self.num_s * self.num_s).swapaxes(1, 2)
        # x: [bs, seq_len, h_dim] -> [bs, h_size, num_s ** 2] -> [bs, num_s ** 2, h_size]

        x = self.sa(x)
        # x: [bs, num_s ** 2, h_size] -> [bs, num_s ** 2, h_size]

        x = x.swapaxes(2, 1).view(-1, self.h_size, self.num_s, self.num_s)
        # x: [bs, num_s ** 2, h_size] -> [bs, h_size, num_s ** 2] -> [bs, seq_len, num_s, num_s]

        return x
    
# =================================================================================================================== #

class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels = None,
                 residual = False
                 ):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = mid_channels, kernel_size = 3, padding = 1, bias = False, stride = 1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(in_channels = mid_channels, out_channels = out_channels, kernel_size = 3, padding = 1, bias = False, stride = 1),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        # x: [bs, ch, h, w]

        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                #  bilinear = True
                 ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual= True),
            DoubleConv(in_channels, out_channels, in_channels//2)
            )

    def forward(self, x):
        return self.maxpool_conv(x)
    
# =================================================================================================================== #

class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bilinear = True
                 ):
        super().__init__()
        self.bilinear =bilinear
        # if bilinear, use the normal conv to reduce the number of channels
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode= 'bilinear', align_corners = True)
            self.conv = DoubleConv(in_channels = in_channels, out_channels = in_channels, residual= True)
            self.conv2 = DoubleConv(in_channels = in_channels, out_channels = out_channels, mid_channels = in_channels //2 )
        else:
            self.up = nn.ConvTranspose2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 2, stride = 2)
            self.conv = DoubleConv(in_channels = in_channels, out_channels = in_channels, residual= False)

    def forward(self, x1, x2):
        # x1: [bs, ch, h, w]
        # x2: [bs, ch', h', w']

        x1 = self.up(x1)
        # x1: [bs, ch, h, w] -> [bs, ch, 2*h, 2*w]

        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2] # h' - 2*h
        diff_x = x2.size()[3] - x1.size()[3] # w' - 2*w

        x1 = F.pad(x1, [diff_x //2, diff_x - diff_x //2, diff_y//2, diff_y - diff_y // 2])
        # x1: [bs, ch, 2*h, 2*w] -> [bs, ch, 2*h + diff_x, 2*w + diff_y]

        x = torch.cat([x2, x1], dim=1)
        # x1: [bs, ch, 2*h + diff_x, 2*w + diff_y]
        # x2: [bs, ch, h', w']

        # if 2*w + diff_y == w':
        # >> x: [bs, ch, 2*h + diff_x + h', w']

        x = self.conv(x)
        if self.bilinear:
            x = self.conv2(x)
        return x

# =================================================================================================================== #

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)
