import argparse
from Datamodule import CEDataModule
from train import TrainClassifier
from models import *

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os
import logging
