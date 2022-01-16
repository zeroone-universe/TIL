import torch
from torch import nn

class DNN(nn.Module):
    def __init__(self,drop_prob=0.5):
        super(DNN,self).__init__()

        self.drop_prob=drop_prob

        self.drop_prob=drop_prob
        self.net = nn.Sequential(
            nn.Linear(3*32*32,256),
	        torch.nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob),
            
            nn.Linear(256,32),
	        torch.nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob),
            
            nn.Linear(32,10)
			)

    def forward(self,x):
        batch_size, _, _, _= x.size()
        x=x.view(batch_size,-1)
        x=self.net(x)
        return x

