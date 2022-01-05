import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F

import random
import numpy as np


class FMNIST_load(pl.LightningDataModule):
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

    '''
    def setup(self,stage):
      
    optional  
    '''
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=128, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=128)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=128)
    
class FMNIST_classifier(pl.LightningModule):
    def __init__(self):
        super(FMNIST_classifier,self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(28*28,256),
	        torch.nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,32),
	        torch.nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,10)
			)
        
    def forward(self,x):
        batch_size, channels, width, height = x.size()
        x=x.view(batch_size,-1)
        x=self.net(x)
        return torch.log_softmax(x, dim=1)

    
    def loss_fn(self, logits, labels):
        return F.nll_loss(logits, labels)

       

    def configure_optimizers(self):
        
        return torch.optim.Adam(model.parameters(),lr=1e-3)
    
    def training_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.loss_fn(y_hat,y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.loss_fn(y_hat,y)
        self.log('val_loss',loss)
        
    
    def test_step(self,batch,batch_dix):
        pass
        #will not be used until I call trainer.test()    
        #return test_loss
    
if __name__=='__main__':
    data_module=FMNIST_load()

    model=FMNIST_classifier()

    trainer=pl.Trainer(gpus=1,
    max_epochs=3,
    progress_bar_refresh_rate=20,
    )
    trainer.fit(model, data_module)