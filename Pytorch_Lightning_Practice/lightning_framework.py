import torch
import torch.
from torch import nn

import pytorch_lightning as pl

class DATASETNAME(pl.LightningDataModule):
    '''
    def prepare_Data(self):
        optional 
    '''
    def setup(self,stage):
        pass
  
    
    
    def train_dataloader(self):
        return 
        
    def val_dataloader(self):
        return 
    
    def test_dataloader(self):
        return 
    
class MODELNAME(pl.LightningModule):
    def __init__(self):
        super(MODELNAME,self).__init__()
        
    def forward(self,x):
        return output
    
    def cal_loss(self, a,b):
        return loss
    
    def configure_optimizers():
        
        return optimizers
    
    def training_step(self,batch,batch_idx):
        
        return loss
    
    def validation_step(self,batch,batch_idx):
        
        return val_loss
    
    def test_step(self,batch,batch_dix):
        #will not be used until I call trainer.test()    
        return test_loss
    
if __name__=='__main__':
    data_module=DATASETNAME()
    trainer=pl.Trainer(gpus=,
    max_epochs=,
    progress_bar_refresh_rate=,
    )
    trainer.fit(MODELNAME, data_module)
    
    
    
