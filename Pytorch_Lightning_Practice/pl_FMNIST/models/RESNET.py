import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock,self).__init__()

        self.conv1=nn.Conv2d(
            in_planes,planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1=nn.BatchNorm2d(planes)
        
        self.conv2=nn.Conv2d(
            planes, planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2=nn.BatchNorm2d(planes)

        self.shortcut=nn.Sequential()

        if stride!=1 or in_planes!=planes:
            self.shortcut=nn.Sequential(
                nn.Conv2d(
                    in_planes, planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes)
            )    

    def forward(self,x):
        x_sc=x
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.bn2(self.conv2(x))
        x+=self.shortcut(x_sc)
        x=F.relu(x)
        return x

class RESNET(nn.Module):
    def __init__(self):
        super(RESNET,self).__init__()

        self.in_planes=16

        self.conv1=nn.Conv2d(1,16,
        kernel_size=3,
        stride=1,
        padding=3,
        bias=False)

        self.bn1=nn.BatchNorm2d(16)
        
        self.layer1=self._make_layer(16,2,stride=1)
        self.layer2=self._make_layer(32,2,stride=2)
        self.layer3=self._make_layer(64,2,stride=2)
        self.linear=nn.Linear(64,10)



    def _make_layer(self,planes, num_blocks, stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes=planes
        return nn.Sequential(*layers)


    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=F.avg_pool2d(x, 8)
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x



if __name__ == '__main__':
    resnet=RESNET()
    print(resnet)