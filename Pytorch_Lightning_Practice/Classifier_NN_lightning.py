import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST("F:\Python_Codes\Data_for_Practice", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=128)

# Initialize a trainer
trainer = Trainer(
    gpus=1,
    max_epochs=3,
    progress_bar_refresh_rate=50,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)