from datamodule import MNIST_datamodule
from train import MNIST_train

from models import *
from utils import *

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import logging

import yaml

def main(config):
    mnist_datamodule = MNIST_datamodule(config)
    mnist_train= MNIST_train(config)
    
    check_dir_exist(config["train"]["logger_path"])
    model_name = config["model"]["model_name"]
    tb_logger=pl_loggers.TensorBoardLogger(config["train"]["logger_path"], name=f"MNIST_{model_name}_logs")

    #callback
    checkpoint_callback = ModelCheckpoint(
    filename = "{epoch}-{val_acc:.2f}-{val_loss:.2f}",
    monitor = "val_acc",
    mode = "max",
    save_top_k = 1
        )
    
    earlystopping = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=config["train"]["earlystop_patience"], verbose=False, mode="max"
        )
    
    
    trainer=pl.Trainer(
        devices=1, accelerator="gpu",
        max_epochs=config["train"]["max_epochs"],
        callbacks=[checkpoint_callback, earlystopping],
        logger=tb_logger
        )
    
    trainer.fit(mnist_train, mnist_datamodule)
    
    trainer.test(mnist_train, mnist_datamodule)
    
if __name__ == "__main__":
    config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    main(config)