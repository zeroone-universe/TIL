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
        #use when training with dataparallel
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

'''
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return prediction
'''