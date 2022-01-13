import argparse
from dataloader import FMNIST_load
from train import TrainClassifier
from models import *

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os

def main(args):
    data_module=FMNIST_load(data_dir=args.data_dir, batch_size=args.batch_size)
    train_classifier=TrainClassifier(model_name=args.model_name, drop_prob=args.drop_prob)

    tb_logger=pl_loggers.TensorBoardLogger(args.logger_path, name=args.loggerdir_name)
    trainer=pl.Trainer(gpus=1,
    max_epochs=100,
    progress_bar_refresh_rate=1,
    callbacks=[EarlyStopping(monitor="val_acc", min_delta=0.00, patience=args.earlystop_patience, verbose=False, mode="max")],
    logger=tb_logger,
    default_root_dir="./"
    )

    trainer.fit(train_classifier, data_module)
    trainer.test(train_classifier, data_module)    



    



if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train FMNIST classifier")
    
    parser.add_argument("--data_dir", default="F:\TIL_Dataset", type=str, help="FMNIST 데이터의 Path")
    parser.add_argument("--batch_size", default=128, type=int, help="배치 사이즈")
    parser.add_argument("--model_name", default='CNN', type=str, help='모델 이름')
    parser.add_argument("--drop_prob", default=0.5, type=int, help='Dropout Probability')
    parser.add_argument("--logger_path", default="F:/TIL/Pytorch_Lightning_Practice/tb_logger/", type=str, help='logger_path')
    parser.add_argument("--loggerdir_name", default='CNN_logs', type=str, help='loggerdir_name')
    parser.add_argument("--earlystop_patience", default=2, type=int, help='earlystop patience')

    args=parser.parse_args()
    main(args)

    


