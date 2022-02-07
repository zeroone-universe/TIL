from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from torch import nn

import sys

import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl

from models.CNN import CNN
from models.DNN import DNN
from models.RESNET import RESNET

from dataloader import CIFAR10_load

class TrainClassifier(pl.LightningModule):
    def __init__(self, args):
        super(TrainClassifier,self).__init__()
        self.drop_prob=args.drop_prob
        
        if args.model_name=="CNN":
            self.classifier=CNN(drop_prob=self.drop_prob)
        elif args.model_name=='DNN':
            self.classifier=DNN(drop_prob=self.drop_prob)
        elif args.model_name=="RESNET":
            self.classifier=RESNET()
        else:
            print("No classifier")
            sys.exit()
            
        print(self.classifier)

        self.train_acc=torchmetrics.Accuracy()
        self.val_acc=torchmetrics.Accuracy()
        self.test_acc=torchmetrics.Accuracy()

    def forward(self,x):
        x=self.classifier(x)
        return x
    
    def loss_fn(self, logits, labels):
        cross_entropy_loss=nn.CrossEntropyLoss()
        return cross_entropy_loss(logits,labels)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(),lr=1e-3)
    
    def training_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.loss_fn(y_hat,y)
        acc=self.train_acc(y_hat,y)
        metrics= {'train_acc':acc, 'train_loss':loss}
        self.log_dict(metrics)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self.forward(x)
        loss=self.loss_fn(y_hat,y)
        acc=self.val_acc(y_hat,y)
        metrics={'val_acc':acc, 'val_loss':loss}
        
        self.log_dict(metrics)
        return loss
    
    def validation_epoch_end(self,outputs):
        pass
    
    def test_step(self,batch,batch_idx):
        
        x,y=batch
        y_hat=self.forward(x)
        loss=self.loss_fn(y_hat,y)
        acc=self.test_acc(y_hat,y)
        metrics={'test_acc':acc, 'test_loss':loss}
        self.log_dict(metrics)

    def predict_step(self, batch, batch_idx):
        pass
        
    
if __name__=='__main__':
    data_module=CIFAR10_load(args)
    model=TrainClassifier(args)
    
    tb_logger = pl_loggers.TensorBoardLogger("F:/TIL/Pytorch_Lightning_Practice/tb_logger/",name='CNN_logs')
    trainer=pl.Trainer(gpus=1,
    max_epochs=100,
    progress_bar_refresh_rate=1,
    callbacks=[EarlyStopping(monitor="val_acc", min_delta=0.00, patience=2, verbose=False, mode="max")],
    logger=tb_logger,
    default_root_dir="./"
    )
    
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    
    fmnist_label={0:'Top', 1:'Trouser', 2:'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

    a=data_module.test_dataloader()

    x, y=next(iter(a))

    y_hat=model(x)
    y_pred=torch.argmax(y_hat,dim=1)
    print(y_pred.shape)

    pltsize=1
    plt.figure(figsize=(10*pltsize, pltsize))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis('off')
        plt.imshow(x[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
        plt.title(f"Pred:{fmnist_label[y_pred[i].item()]}\nLabel:{fmnist_label[y[i].item()]}", fontdict = {'fontsize' : 7})
    plt.show()
    