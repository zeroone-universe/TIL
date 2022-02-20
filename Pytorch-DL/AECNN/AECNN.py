"""
A New Framework for CNN-Based Speech Enhancement in the Time Domain
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AECNN(nn.Module):
    def __init__(self, in_channels=1, out_channels = 1, num_layers = 8, kernel_size=11):
        super().__init__()

        self.name= "AECNN"
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.num_layers = 8 #encoders의 layer 개수  
        self.kernel_size= kernel_size

        self.Encoder = AECNN_Encoder(in_channels, layer_num, kernel_size)
        self.Decoder = AECNN_Decoder(out_channels, layer_num, kernel_size)

    def forward(self, x):
        '''
        while len(x.size()) < 2 :
            x= x.unsqueeze(-2)
        '''

        x_len= x.shape[-1]

        x_in, down = self.Encoder(x)

        x_out = self.Decoder(x, down)

    def get_name(self):
        return self.name

class AECNN_Encoder(nn.Module):
    def __init__(self, in_channels = 1, num_layers = 8, kernel_size = 11):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.down_channels = [2**(6+b//3) for b in range(num_layers)]