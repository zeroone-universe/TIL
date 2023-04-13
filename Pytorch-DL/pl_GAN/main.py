import os

import pytorch_lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import GANTrain
from datamodule import MNIST_datamodule
import yaml

def main(config):
    dm = MNIST_datamodule(config)
    model = GANTrain(config)
    trainer = L.Trainer(
        devices=1, accelerator="gpu",
        max_epochs=100,
    )
    trainer.fit(model, dm)
    
if __name__ == "__main__":
    config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    main(config)