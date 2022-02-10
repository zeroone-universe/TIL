import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import pytorch_lightning as pl


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import UNet
    
class Train_UNet(pl.LightningModule):
    def __init__(self, args):
        super(Train_UNet,self).__init__()
        self.model=UNet(num_classes=args.num_classes)
        self.lr=args.lr


        print(self.model)

    def forward(self,x):
        output=self.model(x)
        return output
        

    def loss_fn(self, logits, labels):
        loss= nn.CrossEntropyLoss()
        return loss(logits, labels)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        return optimizer
    
    def training_step(self,batch,batch_idx):
        x,y= batch
        y_hat=self.forward(x)
        loss=self.loss_fn(y_hat, y)
        self.log("training_loss", loss)
        return loss
    

    def validation_step(self,batch,batch_idx):
        x,y= batch
        y_hat=self.forward(x)
        val_loss=self.loss_fn(y_hat, y)
        self.log("validation_loss", val_loss)
        return val_loss



    def test_step(self,batch,batch_idx):
        #will not be used until I call trainer.test()    
        pass
    
    def iou(y_hat, y):
        pass

    


    
