from os import X_OK
from torch.nn import MSELoss, L1Loss
import torch 
import scipy.signal as sig
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as sig
import numpy as np


class SISNRLoss:
    def __init__(self):
        self.name = "SISNRLoss"

    def __call__(self, x_proc, x_orig):
        length = min(x_proc.shape[-1], x_orig.shape[-1])
        x_proc = x_proc[..., :length].squeeze()
        x_orig = x_orig[..., :length].squeeze()

        x_orig_zm = x_orig - x_orig.mean(dim = -1, keepdim = True)
        x_proc_zm = x_proc - x_proc.mean(dim = -1, keepdim = True)

        x_dot = torch.sum(x_orig_zm * x_proc_zm, dim = -1, keepdim = True)
        
        s_target = x_dot * x_orig_zm / (1e-10 + x_orig_zm.pow(2).sum(dim = -1, keepdim = True))

        e_noise = x_proc_zm - s_target

        SISNR = e_noise.norm(2, dim = -1) + 1e-10

        SISNR = 20*torch.log10(SISNR/(s_target.norm(2, dim = -1)+1e-10))

        return SISNR.mean()

    def get_name(self):
        return self.name

if __name__ == "__main__":
    sisnr_loss = SISNRLoss()
    orig = torch.randn(4, 1, 2048).cuda()
    noise = torch.randn(4, 1, 2048).cuda()

    x_dot = torch.sum(orig*noise)
    '''
    a = sisnr_loss(orig + noise , orig)
    print(a)
    '''