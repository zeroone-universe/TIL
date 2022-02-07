import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F

import argparse



class CIFAR10_load(pl.LightningDataModule):
    def __init__(self,args):
        super().__init__()
        self.data_dir=args.data_dir
        self.batch_size=args.batch_size
        self.transform = ToTensor()
    
    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir,train=True,download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage):
        full_data = datasets.CIFAR10(
        root=self.data_dir, 
        train=True,
        transform=self.transform,
        download=True
        )
        
        self.test_data= datasets.CIFAR10(
        root=self.data_dir, 
        train=False,  
        transform=self.transform,
        download=True
        )

        self.train_data, self.val_data=torch.utils.data.random_split(full_data, [45000,5000])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
    


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Train CNN classifier")
    
    parser.add_argument("--data_dir", default="F:\TIL_Dataset", type=str, help="FMNIST 데이터의 Path")
    parser.add_argument("--batch_size", default=128, type=int, help="배치 사이즈")
    args=parser.parse_args()


    cifar10=CIFAR10_load(args)
    cifar10.setup(stage=None)
    a=cifar10.train_dataloader()
    example_data, example_target=next(iter(a))

    print(f"Example_Data \nDim : {example_data.dim()}, Shape: {example_data.shape}, dtype: {example_data.dtype}, device: {example_data.device} ")
    print(f"Example_target \nDim : {example_target.dim()}, Shape: {example_target.shape}, dtype: {example_target.dtype}, device: {example_target.device} ") 

    fmnist_label={0:'airplane', 1:'automobile', 2:'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    pltsize=1
    plt.figure(figsize=(10*pltsize, pltsize))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis('off')
        plt.imshow(example_data[i,:,:,:].numpy().reshape(3,32,32).swapaxes(0,1).swapaxes(1,2))
        plt.title(fmnist_label[example_target[i].item()])
    plt.show()

        