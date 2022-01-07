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
    
class FMNIST_classifier(pl.LightningModule):
    def __init__(self):
        super(FMNIST_classifier,self).__init__()
        self.drop_prob=0.5
        self.net = nn.Sequential(
            nn.Linear(28*28,256),
	        torch.nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256,32),
	        torch.nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32,10)
			)
        
        self.train_acc=torchmetrics.Accuracy()
        self.val_acc=torchmetrics.Accuracy()
        self.test_acc=torchmetrics.Accuracy()

    def forward(self,x):
        batch_size, _, _, _= x.size()
        x=x.view(batch_size,-1)
        x=self.net(x)
        return x
    
    def loss_fn(self, logits, labels):
        cross_entropy_loss=nn.CrossEntropyLoss()
        return cross_entropy_loss(logits,labels)

    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(),lr=1e-3)
    
    def training_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.loss_fn(y_hat,y)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.loss_fn(y_hat,y)
        acc=self.val_acc(y_hat,y)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc=self.test_acc(y_hat,y)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)     
    
if __name__=='__main__':
    data_module=FMNIST_load()
    model=FMNIST_classifier()

    trainer=pl.Trainer(gpus=1,
    max_epochs=10,
    progress_bar_refresh_rate=20,
    callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.00, patience=2, verbose=False, mode="min")]
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    