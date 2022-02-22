from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from torch import nn

import sys

import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models import *