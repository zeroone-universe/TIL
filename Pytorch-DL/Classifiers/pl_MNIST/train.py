import torch
from torch import nn

import sys

import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl

import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.CNN import CNN
from models.DNN import DNN
# from models.RESNET import RESNET

class MNIST_train(pl.LightningModule):
    def __init__(self, config):
        super(MNIST_train,self).__init__()
        
        
        if config["model"]["model_name"]=="CNN":
            self.classifier=CNN(config)
        elif config["model"]["model_name"]=='DNN':
            self.classifier=DNN(config)
        elif config["model"]["model_name"]=="RESNET":
            self.classifier=RESNET(config)
        else:
            print("No classifier")
            sys.exit()
            
        print(self.classifier)
        
        self.train_acc=torchmetrics.classification.MulticlassAccuracy(num_classes = 10)
        self.val_acc=torchmetrics.classification.MulticlassAccuracy(num_classes = 10)
        self.test_acc=torchmetrics.classification.MulticlassAccuracy(num_classes = 10)
        
    def forward(self,x):
        output = self.classifier(x)
        return output
        #forward defines the prediciton/inference actions
        
    def criterion(self, predict, target):
        return F.cross_entropy(predict, target)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(),lr=1e-3)
    
    def training_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        acc = self.train_acc(y_hat, y)
        metrics= {'train_acc':acc, 'train_loss':loss}
        self.log_dict(metrics, prog_bar = True)
        return loss
        
    def validation_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.criterion(y_hat,y)
        acc=self.val_acc(y_hat,y)
        self.log('val_acc', acc, prog_bar = True)
        self.log('val_loss', loss, prog_bar = True)
        return loss
    
    def test_step(self,batch,batch_idx):
        
        x,y=batch
        y_hat=self.forward(x)
        loss=self.criterion(y_hat,y)
        acc=self.test_acc(y_hat,y)
        metrics={'test_acc':acc, 'test_loss':loss}
        self.log_dict(metrics)
        print(f"Test Accuracy: {acc}, Test Loss: {loss}")

    def predict_step(self, batch, batch_idx):
        pass
    
