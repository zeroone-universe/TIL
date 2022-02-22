"""
A New Framework for CNN-Based Speech Enhancement in the Time Domain
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AECNN(nn.Module):
    def __init__(self, in_channels=1, out_channels = 1, num_layers = 8, kernel_size=11):
        super(AECNN, self).__init__()

        self.name= "AECNN"
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.num_layers = num_layers #encoder과 decoder의 layer number
        self.kernel_size= kernel_size

        self.Encoder = AECNN_Encoder(in_channels, num_layers, kernel_size)
        self.Decoder = AECNN_Decoder(out_channels, num_layers, kernel_size)

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

        self.unet_down = nn.ModuleList([UNet_down(
            in_channels= self.down_channels[l-1] if l>0 else in_channels,
            out_channels=self.down_channels[l],
            kernel_size=kernel_size,
            stride=2 if l>0 else 1,
            dropout=0.2 if l%3 ==2 else 0,
            bias=True,
        ) for l in range(self.num_layers)])

        self.unet_bottle = UNet_down(
            in_channels= 2**(6+(num_layers -1)//3),
            out_channels= 2**(6+num_layers//3),
            kernel_size= kernel_size,
            bias=True,
            stride=2,
            dropout=0.2 if num_layers%3==2 else 0,
        )

    def forward(self,x):
        '''
        while len(x.size())< 2:
            x=x.unsqueeze(-2)
        '''

        down = []
        for l in range(self.num_layers):
            x=self.unet_down[l](x)
            down.append(x)

        x= self.unet_bottle(x)

        return x, down

class AECNN_Decoder(nn.Module):
    def __init__(self, out_channels=1, num_layers= 8 , kernel_size = 11):
        super.__init__()

        self.out_channels= out_channels
        self.num_layers = num_layers
        down_channels = [2**(6+b//3) for b in range(self.num_layers)]
        up_channels = list(reversed(down_channels))

        self.unet_up = ([UNet_up])




class UNet_down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0, bias= True):
        super().__init__()        
        self.conv = nn.Conv1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              dilation = 1, bias=bias, padding =kernel_size//2)
        #nn.init.orthogonal_(self.conv.weight)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout = dropout
        
        self.actiㄹvation = nn.LeakyReLU()
        
        if dropout>0:
            self.do = nn.Dropout(dropout)
        
        # self.BN = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        l = x.shape[-1]
        x = F.pad(x, pad=(0, self.kernel_size))
        x = self.conv(x)
        
        x = x[..., :l//self.stride]
        #x = x[..., :l//self.stride+1]
        
        x = self.activation(x)

        if self.dropout:
            x = self.do(x)
                
        print(x.shape)
        return x
        
if __name__ == '__main__':
    model = AECNN_Encoder().cuda()
    
    
    
    model.train()
    x = th.randn(4, 1, 2048).cuda()
    y = th.randn(4, 1, 32000).cuda()
    z = model(x)

    #print(count_params(model))
    del model