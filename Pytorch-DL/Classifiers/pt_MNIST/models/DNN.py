import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, config):
        super(DNN,self).__init__()
        
        self.drop_prob = config["model"]["drop_prob"]
        
        self.net = nn.Sequential(
            nn.Linear(28*28, 256),
            torch.nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p = self.drop_prob),
            
            nn.Linear(256, 32),
            torch.nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p = self.drop_prob),
            
            nn.Linear(32,10),
            nn.LogSoftmax(dim = 1)
        )
        
    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1)
        x = self.net(x)
        return x
    
    