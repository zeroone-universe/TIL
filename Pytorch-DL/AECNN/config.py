#-----------------------------------------------
#Config that does not have impact on performance
#-----------------------------------------------

RANDOM_SEED = 0b011011


#-----------------------------------------------
#Training
#-----------------------------------------------

EPOCHS_SAVE_START = 0
OUTPUT_DIR_PATH = "/media/youngwon/Neo/NeoChoi/TIL/TIL_Dataset/AECNN_enhancement/TIMIT_enhanced"
LOGGER_PATH = "/media/youngwon/Neo/NeoChoi/TIL/Pytorch-DL/AECNN/tb_logger"

MAX_EPOCHS= 30

#-----------------------------------------------
#1. Dataset
#-----------------------------------------------
#directory that have every dataset in it.
DATA_DIR = "/media/youngwon/Neo/NeoChoi/TIL/TIL_Dataset/AECNN_enhancement"

INPUT_DIR = "TIMIT_decoded"
TARGET_DIR = "TIMIT"

BATCH_SIZE = 4
SEG_LEN = 2


#-----------------------------------------------
#2. Model
#-----------------------------------------------
WINDOW_SIZE = 2048
HOP_SIZE = 512

NUM_LAYERS = 8
KERNEL_SIZE = 11

#-----------------------------------------------
#3. Loss
#-----------------------------------------------
LOSS_TYPE = "STFTLoss"
#LOSS_TYPE = "SISNRLoss"
#for STFT Loss
STFTLOSS_WINDOW_SIZE = 512
STFTLOSS_HOP_SIZE = 256

#-----------------------------------------------
#4. Optimizer
#-----------------------------------------------
INITIAL_LR = 0.001
LR_GAMMA = 0.9
