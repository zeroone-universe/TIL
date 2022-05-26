import argparse
from Datamodule import CEDataModule
from train import TrainAECNN

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import *

def main(args):
    pl.seed_everything(args.seed, workers=True)

    ce_datamodule = CEDataModule(data_dir = args.data_dir, batch_size = args.batch_size, seg_len = args.seg_len)
    train_aecnn = TrainAECNN(args)
    
    check_dir_exist(args.logger_path)
    tb_logger = pl_loggers.TensorBoardLogger(args.logger_path, name=f"AECNN_logs")

    trainer=pl.Trainer(gpus=1,
    max_epochs=args.max_epochs,
    progress_bar_refresh_rate=1,
    callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.earlystop_patience, verbose=False, mode="min")],
    logger=tb_logger,
    default_root_dir="./"
    )

    trainer.fit(train_aecnn, ce_datamodule)







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Train AECNN")
    
    #setting args
    parser.add_argument("--seed", default = 0b011011, type = int, help = "random_seed")
    
    #dataloader args
    parser.add_argument("--data_dir", default="/media/youngwon/Neo/NeoChoi/TIL_Dataset/AECNN_enhancement", type = str, help = "data_dir")
    parser.add_argument("--batch_size", default = 4, type = int, help = "batch_size")
    parser.add_argument("--seg_len", default = 2,  type = int, help = "seg_len")

    #model args
    parser.add_argument("--lr", default = 0.0001)

    #training_args
    parser.add_argument("--logger_path", default = "/media/youngwon/Neo/NeoChoi/TIL/Pytorch-DL/AECNN/tb_logger", type = str, help = "logger_path")
    parser.add_argument("--max_epochs", default = 100, type = int, help = "max_epochs")
    parser.add_argument("--earlystop_patience", default = 3, type = int, help = "earlystop patience")
    parser.add_argument("--loss_type", default = "SISNR", type = str, help = "loss type")

    args = parser.parse_args()
    main(args)