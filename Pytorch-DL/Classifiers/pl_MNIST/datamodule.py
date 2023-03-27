import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F

class MNIST_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        self.data_dir = config["datamodule"]["data_dir"]
        self.batch_size = config["datamodule"]["batch_size"]
        self.transform = transforms.ToTensor()
        
        #???
        self.prepare_data_per_node = True
        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = False
        
    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage):
        
        mnist_full = datasets.MNIST(self.data_dir, train = True, transform = self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, (55000, 5000))
        
    
        self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform = self.transform)
        self.mnist_predict = datasets.MNIST(self.data_dir, train=False, transform = self.transform)
            
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = self.batch_size, shuffle = True, num_workers= 16)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size = self.batch_size, shuffle = True, num_workers= 16)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size = self.batch_size, num_workers= 16)
    
    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=1, num_workers= 16)
