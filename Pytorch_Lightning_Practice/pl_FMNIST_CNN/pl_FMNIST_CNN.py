import torch
from torch import nn

import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
    

class CNN(pl.LightningModule):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.fc1=nn.Sequential(
            nn.Linear(128*4*4,4*128),
            nn.ReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(4*128,10)
        )

        self.train_acc=torchmetrics.Accuracy()
        self.val_acc=torchmetrics.Accuracy()
        self.test_acc=torchmetrics.Accuracy()

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)
    
    def cal_loss(self, logits,label):
        
        return F.nll_loss(logits,label)
    
    def configure_optimizers(self):
        
        return torch.optim.Adam(model.parameters(),lr=1e-3)
    
    def training_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.cal_loss(y_hat,y)
        acc=self.train_acc(y_hat,y)
        metrics={'train_acc':acc, 'train_loss':loss}
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.cal_loss(y_hat,y)
        acc=self.val_acc(y_hat,y)
        metrics={'val_acc':acc, 'val_loss':loss}
        
        self.log_dict(metrics)
        return loss
    
    def validation_epoch_end(self,outputs):
        pass      
    
    def test_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.cal_loss(y_hat,y)
        acc=self.test_acc(y_hat,y)
        metrics={'test_acc':acc, 'test_loss':loss}
        self.log_dict(metrics)
        
    
if __name__=='__main__':
    data_module=FMNIST_load()
    model=CNN()
    
    tb_logger = pl_loggers.TensorBoardLogger("F:/TIL/Pytorch_Lightning_Practice/tb_logger/")
    trainer=pl.Trainer(gpus=1,
    max_epochs=10,
    progress_bar_refresh_rate=1,
    callbacks=[EarlyStopping(monitor="val_acc", min_delta=0.00, patience=3, verbose=False, mode="max")],
    logger=tb_logger
    )
    
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    
    