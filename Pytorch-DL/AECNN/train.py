import pytorch_lightning as pl
import torch
from torch import nn

import sys

import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.AECNN_ek import AECNN
from Loss import *

class TrainAECNN(pl.LightningModule):
    def __init__(self, args):
        super(TrainAECNN, self).__init
        self.model = AECNN()
        self.lr = args.lr
        
    def forward(self,x):
        output= self.model(x)
        
        return output

    def loss_fn(self, s_noisy, s_orig):
        if args.loss_type == "SISNR":
            loss_function = SISNRLoss()
        
        return loss_function(s_noisy, s_orig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
        
    def training_step(self, batch, batch_idx):
        wav_dec, wav_orig = batch
        wav_enh = self.forward(wav_dec)
        loss = self.loss_fn(wav_enh, wav_orig)
        self.log("training_loss" , loss)
        return loss 

    def validation_step(self, batch, batch_idx):
        wav_dec, wav_orig = batch
        wav_enh = self.forward(wav_dec)
        val_loss = self.loss_fn(wav_enh, wav_orig)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass