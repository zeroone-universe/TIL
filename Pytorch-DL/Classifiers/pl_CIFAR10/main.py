import argparse
from dataloader import CIFAR10_load
from train import TrainClassifier
from models import *

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os
import logging
'맥 옮긴김에 테스트'


def main(args):
    print(args.model_name)
    pl.seed_everything(args.seed, workers=True)
    
   
    data_module=CIFAR10_load(args)
    
    train_classifier=TrainClassifier(args)

    tb_logger=pl_loggers.TensorBoardLogger(args.logger_path, name=f"CIFAR10_{args.model_name}_logs")
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
    
    #dataloader args
    parser.add_argument("--data_dir", default="/media/youngwon/NeoChoi/NeoChoi/TIL_Dataset", type=str, help="FMNIST 데이터의 Path")
    parser.add_argument("--batch_size", default=128, type=int, help="배치 사이즈")

    #train arges
    parser.add_argument("--model_name", default='RESNET', type=str, help='모델 이름')
    parser.add_argument("--drop_prob", default=0.5, type=int, help='Dropout Probability')
    parser.add_argument("--logger_path", default="/media/youngwon/NeoChoi/NeoChoi/TIL/DeepLearning/Classifiers/tb_logger", type=str, help='logger_path')
    parser.add_argument("--earlystop_patience", default=2, type=int, help='earlystop patience')
    parser.add_argument("--seed", default=0b011011, type=int, help='random seed')

    args=parser.parse_args()
    main(args)

    


