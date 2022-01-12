import torch
from torch import nn

import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F

import random
import numpy as np


class FMNIST_load(pl.LightningDataModule):
    '''
    def prepare_Data(self):
        optional 
    '''
    
    def setup(self, stage):
        full_data = datasets.FashionMNIST(
        root="F:\Python_Codes\Data_for_Practice", 
        train=True, 
        download=True, 
        transform=ToTensor()
        )
        
        self.test_data= datasets.FashionMNIST(
        root="F:\Python_Codes\Data_for_Practice", 
        train=False, 
        download=True, 
        transform=ToTensor()
        )

        self.train_data, self.val_data=torch.utils.data.random_split(full_data, [50000,10000])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=128, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=128)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=128)

if __name__=='__main__':
    fmnist=FMNIST_load()
    fmnist.setup(stage=None)
    a=fmnist.train_dataloader()
    #x,y=next(iter(train_dataloader))
    
    for x,y in a:
        
        print("Shape of X: ", x.shape, type(x))
        print("Shape of y: ", y.shape, y.dtype)
        img=x[0].squeeze()
        print(f"Shape of img: {x[0].shape} to {img.shape}")
        label=y[0]
        plt.imshow(img,cmap='gray')
        plt.show()
        print(f"Label: {label}")
        break
        
        
