import torch
from torch import nn

import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

import pytorch_lightning as pl

    
class MODELNAME(pl.LightningModule):
    def __init__(self):
        super(MODELNAME,self).__init__()
        
    def forward(self,x):
        return output
        #forward defines the prediciton/inference actions
    def cal_loss(self, a,b):
        return loss
    
    def configure_optimizers(self):
        
        return optimizers
    
    def training_step(self,batch,batch_idx):
        
        return loss
    
    '''
    def training_step_end(self, batch_parts)
        #use when training with datapara  llel
        #training_step의 return 받는다. 
        $Subbatch 있을때만 쓰면 될 듯? 거의 쓸일 없다 보면 될 것 같다.
        return loss
    '''
    
   
    def training_epoch_end(self, training_step_outputs)
        #training_step 혹은 training_step_end의 아웃풋들을 리스트로 받는다.
    

    def validation_step(self,batch,batch_idx):
        return val_loss

    '''
    def validation_step_end(self, batch_parts):
        return something 
    '''

    def validation_epoch_end(self,validation_step_outputs):
        return 

    def test_step(self,batch,batch_idx):
        #will not be used until I call trainer.test()    
        return test_loss
    
    def test_epoch_end(self, test_step_outputs):
        return something

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return a

if __name__=='__main__':
    data_module=DATASETNAME()
    
    tb_logger = pl_loggers.TensorBoardLogger("어디어디logs/")
    trainer=pl.Trainer(
        logger=tb_logger,
        gpus=,
        max_epochs=,
        progress_bar_refresh_rate=,
        )
    trainer.fit(MODELNAME, data_module)
    trainer.test()
    
    
