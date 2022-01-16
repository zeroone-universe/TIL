import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self,drop_prob=0.5):
        super(CNN,self).__init__()
        self.drop_prob=drop_prob

        self.conv1=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1=nn.Sequential(
            nn.Linear(128*4*4,4*128),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob),
        )
        self.fc2=nn.Sequential(
            nn.Linear(4*128,10)
        )


    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x


