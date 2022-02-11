import os
from tkinter.tix import Y_REGION
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
        
        self.batch_size= args.batch_size

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

        iou_val=self.iou(y_hat, y)
        self.log("validation_loss", val_loss) 
        self.log("validation_iou", iou_val)



    def test_step(self,batch,batch_idx):
        #will not be used until I call trainer.test()    
        pass

    def predict_step(self,batch, batch_idx):
        if batch_idx==0:
            
            inverse_transform = transforms.Compose([
                transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
            ])

            x,y = batch
            y_hat=self.forward(x)
            y_pred=torch.argmax(y_hat, dim=1)
            print(f"y_hat shape: {y_hat.shape}, y_pred shape: {y_pred.shape}")
    
            fix, axes= plt.subplots(self.batch_size, 3, figsize=(3*5, self.batch_size*5))
            for i in range(self.batch_size):
                landscape=inverse_transform(x[i]).permute(1,2,0).cpu().numpy()
                label_class=y[i].cpu().numpy()
                label_class_predicted=y_pred[i].cpu().numpy()

                axes[i, 0].imshow(landscape)
                axes[i, 0].set_title("Landscape")
                axes[i, 1].imshow(label_class)
                axes[i, 1].set_title("Label Class")
                axes[i, 2].imshow(label_class_predicted)
                axes[i, 2].set_title("Label Class - Predicted")
            plt.savefig('result.png', dpi=300)

    def iou(self, y_hat, y):
        y_pred=torch.argmax(y_hat, dim=1)
        label_class = y.cpu().numpy()
        label_class_predicted =y_pred.cpu().numpy()
        intersection = np.logical_and(label_class, label_class_predicted)
        union = np.logical_or(label_class, label_class_predicted)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    


    
