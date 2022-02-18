# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 13:50:04 2021

@author: LEE
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AECNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, B=8, kernel_size = 11):
        super().__init__()
        
        self.name = 'AECNN'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.B = B
        self.kernel_size = kernel_size
        
        self.Encoder = AECNN_Encoder(in_channels, B, kernel_size)
        self.Decoder = AECNN_Decoder(out_channels, B, kernel_size)
        
    def forward(self, x):
        while len(x.size()) < 2:
            x = x.unsqueeze(-2)

        x_len = x.shape[-1]
        
        x, down = self.Encoder(x)
        
        print('2222',x.shape)
        print(len(down))
        
        x1 = self.Decoder(x, down)
        print('333',x1.shape)
        
        x2 = x1[:, :, :x_len] # == [..., :x_len]
        print('444',x2.shape)
        
        x = self.Decoder(x, down)[..., :x_len]
        
        
        
        

        return x

    def get_name(self):
        return self.name

class AECNN_Encoder(nn.Module):
    def __init__(self, in_channels = 1, num_layers= 8 , kernel_size = 11):
        super().__init__()
        
        self.in_channels = in_channels
        self.B = B = num_layers
        self.down_channels   = down_channels = [2**(6+b//3) for b in range(num_layers)]           #out_channels
        
        
        self.unet_down = nn.ModuleList([UNet_down(
                                                  in_channels=down_channels[l-1] if l > 0 else in_channels, 
                                                  out_channels=down_channels[l],
                                                  kernel_size=kernel_size,
                                                  stride=2 if l > 0 else 1, 
                                                  dropout = 0.2 if l%3 == 2 else 0,
                                                  bias= True
                                                  ) for l in range(B)])
        
        self.unet_bottle = UNet_down(in_channels= 2**(6+(B-1)//3),
                                     out_channels=  2**(6+B//3),
                                     kernel_size = kernel_size,                                                  
                                     bias=  True,
                                     stride=2, dropout = 0.2 if B%3 == 2 else 0)

    def forward(self, x):
        while len(x.size()) < 2:
            x = x.unsqueeze(-2)
        
        # down skip connectoin 
        down = []
        for b in range(self.B):
            x = self.unet_down[b](x)
            down.append(x)
            print(x.shape)
        x = self.unet_bottle(x)
        print(x, len(down))
        return x, down
    
class AECNN_Decoder(nn.Module):
    def __init__(self, out_channels=1, num_layers= 8, kernel_size = 11):
        super().__init__()
        
        self.out_channels = out_channels
        self.B = B = num_layers
        down_channels   = [2**(6+b//3) for b in range(num_layers)]           #out_channels
        up_channels     = list(reversed(down_channels))             #out_channels

        self.unet_up = nn.ModuleList([UNet_up(
                                                  in_channels=down_channels[-l]+up_channels[l-1] if l > 0 else down_channels[-1], 
                                                  out_channels=up_channels[l]*2,
                                                  kernel_size=kernel_size,
                                                  stride=1, activation = 'prelu', 
                                                  dropout = 0.2 if l%3 == 2 else 0,
                                                  bias= True, r = 2
                                                  ) for l in range(B)])    
        
        self.unet_final = UNet_up(in_channels = down_channels[0]+up_channels[-1],
                                  out_channels = out_channels,
                                  kernel_size = kernel_size,
                                  stride = 1, activation=None, dropout = 0, bias=True, r = 1
                                  )
                
    def forward(self, x, down):

        for b in range(self.B):
            x = self.unet_up[b](x, down[-b-1])

        x = self.unet_final(x, None)
        return x
        
class AECNN_coupling(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        
        
        self.B = B = num_layers
        down_channels   = [2**(6+b//3) for b in range(num_layers)]           #out_channels
        bn_channels     = 2**(6+B//3)
    
        self.down_layers = nn.ModuleList([nn.Conv1d(   in_channels = down_channels[l], 
                                                       out_channels = down_channels[l], 
                                                       kernel_size =1 , stride= 1) for l in range(B)])

        self.bn_layer = nn.Conv1d(  in_channels = bn_channels, 
                                    out_channels = bn_channels, 
                                    kernel_size =1 , stride= 1)

    def forward(self, x, down):
        for b in range(self.B):
            down[b] = self.down_layers[b](down[b])
            print(down[b].shape)
            
        x = self.bn_layer(x)

        return x, down

class UNet_down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0, bias= True):
        super().__init__()        
        self.conv = nn.Conv1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              dilation = 1, bias=bias, padding =kernel_size//2)
        nn.init.orthogonal_(self.conv.weight)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout = dropout
        
        self.activation = nn.LeakyReLU()
        
        if dropout>0:
            self.do = nn.Dropout(dropout)
        
        # self.BN = nn.BatchNorm1d(out_channels)
        
        
        
    def forward(self, x):
        l = x.shape[-1]
        x = F.pad(x, pad=(0, self.kernel_size))
        x = self.conv(x)
        
        x = x[..., :l//self.stride+1]
        
        
        x = self.activation(x)

        if self.dropout:
            x = self.do(x)
                
        
        return x
        
class UNet_up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='leaky_relu', dropout=0, bias= True, r = 2):
        super().__init__()

        self.conv = nn.Conv1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding=kernel_size//2, bias=bias)
        nn.init.orthogonal_(self.conv.weight)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size 
        self.stride = stride
        self.dropout = dropout
        self.r = r
        if dropout>0:
            self.do = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU() if activation != None else activation
        self.DimShuffle = PixelShuffle1D(r)
        

    def forward(self, x, x_prev):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)

        if self.dropout:
            x = self.do(x)                  
            
        x = self.DimShuffle(x)
        
        if x_prev is not None:
            x = th.cat([x[..., :x_prev.shape[-1]], x_prev], dim=1)
        return x
        
class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width
        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        return x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
def count_params(model):
 return sum([param.numel() if param.requires_grad==True else 0 for param in model.parameters()])
 
     
if __name__ == '__main__':
    model = AECNN().cuda()
    
    non_linear = AECNN_coupling()
    
    
    
    print(model.Encoder)
    print(non_linear)
    model.train()
    x = th.randn(4, 1, 32000).cuda()
    y = th.randn(4, 1, 32000).cuda()
    z = model(x)

    # #print(count_params(model))
    del model