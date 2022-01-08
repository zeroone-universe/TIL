from logging import logProcesses
import torch
from torch import nn

import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

import pytorch_lightning as pl

class FMNIST_load(pl.LightningDataModule):
    '''
    def prepare_Data(self):
        optional 
    '''
    
    def setup(self, stage):
        self.train_data = datasets.FashionMNIST(
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

    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=128, shuffle=True)
    
    def val_dataloader(self):
        pass
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=128)

class Generator(nn.module):
    def __init__(self, input_dim=100, img_shape=(28,28)):
        super(generator,self).__init__()
        self.input_dim=input_dim
        self.img_shape=img_shape
        
        def block(in_feat, out_feat):
            layers=[nn.Linear(in_feat, out_feat),
            nn.BatchNorm1d(out_feat),
            nn.ReLU()]
            return layers

        self.model=nn.Sequential(
            *block(self.input_dim,128),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self,z):
        img=self.model(z)
        img=img.view(img.size(0),*self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(28,28)):
        super(Discriminator,self).__init__()
    
        self.model=nn.Sequential(
            nn.Linear(int(np.prod(img_shape)),512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,img):
        img=img.view(img.size(0),-1)
        out=self.model(img)
        return out

class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN,self).__init__()
        self.Generator=Generator()
        self.Discriminator=Discriminator()

    def forward(self,x):
        return self.Generator(z)
    
    def cal_loss(self, a,b):
        return F.binary_cross_entropy(y_hat, y)
    
    def configure_optimizers(self):
        opt_g= torch.optim.Adam(self.Generator.parameters()lr=1e-3)
        opt_d=torch.optim.Adam(self.Discriminator.parameters(),lr=1e-3)
        return [opt_g, opt_d]
    
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
        pass

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
    
    
