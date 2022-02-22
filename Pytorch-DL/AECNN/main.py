import argparse
from Datamodule import CEDataModule
from train import TrainAECNN

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os
import logging

def main(args):




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Train AECNN")
    
    #setting args
    parser.add_argument("--seed", default = 0b011011, type = int, help = "random_seed")
    
    #dataloader args
    parser.add_argument("--data_dir", default="/media/youngwon/Neo/NeoChoi/TIL_Dataset/AECNN_enhancement", type = str, help = "data_dir")
    parser.add_argument("--batch_size", default = 4, type = int, help = batch_size)
    parser.add_argument("--seg_len", default = 2,  type = int, help = seg_len)

    #model args
    parser.add_argument("--lr", default = 1)