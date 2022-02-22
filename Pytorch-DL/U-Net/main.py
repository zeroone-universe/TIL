import argparse
from train import Train_UNet
from Datamodule import CityscapeDataset

from pytorch_lightning import loggers 
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset, DataLoader

import os
import logging

from sklearn.cluster import KMeans
import numpy as np

def main(args):
    pl.seed_everything(args.seed, workers=True)

    
    color_array = np.random.choice(range(256), 3*1000).reshape(-1, 3)
    label_model = KMeans(n_clusters = 10)
    label_model.fit(color_array)
    
    train_dataset=CityscapeDataset(image_dir=f"{args.data_dir}/train", label_model=label_model)
    validation_dataset=CityscapeDataset(image_dir=f"{args.data_dir}/val", label_model=label_model)

    train_dataloader=DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    validation_dataloader=DataLoader(dataset=validation_dataset, batch_size=args.batch_size)

    train_UNet=Train_UNet(args)

    tb_logger=loggers.TensorBoardLogger(args.logger_path, name=f"UNet_logs")
    trainer=pl.Trainer(gpus=1,
    max_epochs=args.max_epochs,
    progress_bar_refresh_rate=1,
    callbacks=[EarlyStopping(monitor="validation_loss", min_delta=0.00, patience=args.earlystop_patience, verbose=False, mode="min")],
    logger=tb_logger,
    default_root_dir="./"
    )

    trainer.fit(train_UNet, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
 
    trainer.predict(dataloaders=validation_dataloader) 



    



if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train UNet")
    #setting args
    parser.add_argument("--seed", default=0b011011, type=int, help='random seed')


    #dataloader args
    parser.add_argument("--data_dir", default="/media/youngwon/Neo/NeoChoi/TIL_Dataset/cityscapes_data", type=str, help="FMNIST 데이터의 Path")
    parser.add_argument("--batch_size", default=8, type=int, help="배치 사이즈")

    #model args
    parser.add_argument("--lr", default=0.01, type=float, help='logger_path')
    parser.add_argument("--num_classes", default=10, type=int, help='logger_path')

    #training args
    parser.add_argument("--logger_path", default="/media/youngwon/Neo/NeoChoi/TIL/Pytorch-DL/U-Net/tb_logger", type=str, help='logger_path')
    parser.add_argument("--max_epochs", default=100, type=int, help='max_epochs')
    parser.add_argument("--earlystop_patience", default=2, type=int, help='earlystop patience')
    
    
    args=parser.parse_args()
    main(args)
