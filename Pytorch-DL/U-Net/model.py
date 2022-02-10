from sklearn.feature_selection import SequentialFeatureSelector
import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet,self).__init__()
        self.contracting_layer1=self.ConvBlock(3,64) #[-1,64,256,256]
        self.maxpool1=nn.MaxPool2d(kernel_size=2, stride=2) #[-1,64,128,128]
        self.contracting_layer2=self.ConvBlock(64,128) #[-1,128,128,128]
        self.maxpool2=nn.MaxPool2d(kernel_size=2, stride=2) #[-1,128,64,64]
        self.contracting_layer3=self.ConvBlock(128,256)
        self.maxpool3=nn.MaxPool2d(kernel_size=2, stride=2) #[-1,256,32,32]
        self.contracting_layer4=self.ConvBlock(256,512) #[-1,512,32,32]
        self.maxpool4=nn.MaxPool2d(kernel_size=2, stride=2) #[-1,512,16,16]
        self.middle_layer=self.ConvBlock(512,1024) #[-1,1024,16,16]
        self.upconv1=nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1) #[-1,512,32,32]
        self.expand_layer1=self.ConvBlock(1024, 512) #[-1,1024,32,32]->[-1,512,32,32]
        self.upconv2=nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1) #[-1,256,64,64]
        self.expand_layer2=self.ConvBlock(512, 256) #[-1,512,64,64]->[-1,256,64,64]
        self.upconv3=nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1) #[-1,128,128,128]
        self.expand_layer3=self.ConvBlock(256, 128) #[-1,256,128,128]->[-1,128,128,128]
        self.upconv4=nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) #[-1,64,256,256]
        self.expand_layer4=self.ConvBlock(128, 64) #[-1,128,256,256]->[-1,64,256,256]
        self.output=nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

    def ConvBlock(self, in_channels, out_channels):
        block=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        )
        return block

    def forward(self,inp):
        contract_11 = self.contracting_layer1(inp) #[-1,64,256,256]
        contract_12 = self.maxpool1(contract_11) #[-1,64,128,128]
        contract_21 = self.contracting_layer2(contract_12) #[-1,128,128,128]
        contract_22 = self.maxpool2(contract_21)
        contract_31 = self.contracting_layer3(contract_22) #[-1,256,6,64]
        contract_32 = self.maxpool3(contract_31)
        contract_41 = self.contracting_layer4(contract_32) #[-1,512,32,32]
        contract_42 = self.maxpool4(contract_41) #[-1,512,16,16]
        mid = self.middle_layer(contract_42)
        expand_11=self.upconv1(mid)  #[-1,512,32,32]
        expand_12=self.expand_layer1(torch.cat((expand_11,contract_41),dim=1))
        expand_21=self.upconv2(expand_12)
        expand_22=self.expand_layer2(torch.cat((expand_21,contract_31),dim=1))
        expand_31=self.upconv3(expand_22)
        expand_32=self.expand_layer3(torch.cat((expand_31,contract_21),dim=1))
        expand_41=self.upconv4(expand_32)
        expand_42=self.expand_layer4(torch.cat((expand_41,contract_11),dim=1))
        output= self.output(expand_42)


        return output

if __name__=="__main__":
    unet=UNet(num_classes=10)
    print(unet)
    x=torch.rand(12,3,256,256)
    output=unet(x)
    print(output)
    print(output.shape)