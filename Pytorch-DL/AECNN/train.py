import pytorch_lightning as pl
import torch
from torch import nn

import sys

import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.AECNN_01 import AECNN
from Loss import *

from pesq import pesq

class TrainAECNN(pl.LightningModule):
    def __init__(self, args):
        super(TrainAECNN, self).__init__()
        self.model = AECNN()
        self.lr = args.lr

        self.loss_type =  args.loss_type
        
    def forward(self,x):
        output= self.model(x)
        
        return output

    def loss_fn(self, s_noisy, s_orig):
        if self.loss_type == "SISNR":
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

        wav_enh = wav_enh.squeeze().cpu().numpy()
        wav_orig = wav_orig.squeeze().cpu().numpy()

        val_pesq = pesq(fs = 16000, ref = wav_orig, deg = wav_enh, mode = "wb")
        self.log("val_loss", val_loss)
        self.log("val_pesq", val_pesq)
        
    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass