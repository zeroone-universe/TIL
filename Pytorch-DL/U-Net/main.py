import argparse
from train import Train_UNet
from Datamodule import CityscapeDatamodule

from pytorch_lightning import loggers 
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os
import logging

def main(args):
    pl.seed_everything(args.seed, workers=True)
    
   
    data_module=CityscapeDatamodule(args)
    
    train_UNet=Train_UNet(args)

    tb_logger=loggers.TensorBoardLogger(args.logger_path, name=f"UNet_logs")
    trainer=pl.Trainer(gpus=1,
    max_epochs=10,
    progress_bar_refresh_rate=1,
    #callbacks=[EarlyStopping(monitor="val_acc", min_delta=0.00, patience=args.earlystop_patience, verbose=False, mode="max")],
    logger=tb_logger,
    default_root_dir="./"
    )

    trainer.fit(train_UNet, data_module)
 
    #trainer.test(train_classifier, data_module)    



    



if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train FMNIST classifier")
    #setting args
    parser.add_argument("--seed", default=0b011011, type=int, help='random seed')


    #dataloader args
    parser.add_argument("--data_dir", default="/media/youngwon/Neo/NeoChoi/TIL_Dataset/cityscapes_data", type=str, help="FMNIST 데이터의 Path")
    parser.add_argument("--batch_size", default=8, type=int, help="배치 사이즈")

    #model&train args
    parser.add_argument("--lr", default=0.01, type=float, help='logger_path')
    parser.add_argument("--num_classes", default=10, type=int, help='logger_path')
    parser.add_argument("--logger_path", default="/media/youngwon/Neo/NeoChoi/TIL/Pytorch-DL/U-Net/tb_logger", type=str, help='logger_path')

    '''
    parser.add_argument("--earlystop_patience", default=2, type=int, help='earlystop patience')
    parser.add_argument("--seed", default=0b011011, type=int, help='random seed')
    '''
    args=parser.parse_args()
    main(args)
