import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F



class FMNIST_load(pl.LightningDataModule):
    def __init__(self,data_dir="F:\TIL_Dataset", batch_size=128):
        super().__init__()
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.transform = ToTensor()
    
    def prepare_data(self):
        datasets.FashionMNIST(root=self.data_dir,train=True,download=True)
        datasets.FashionMNIST(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage):
        full_data = datasets.FashionMNIST(
        root=self.data_dir, 
        train=True,
        transform=self.transform,
        download=True
        )
        
        self.test_data= datasets.FashionMNIST(
        root=self.data_dir, 
        train=False,  
        transform=self.transform,
        download=True
        )

        self.train_data, self.val_data=torch.utils.data.random_split(full_data, [50000,10000])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
    


if __name__=='__main__':
    fmnist=FMNIST_load()
    fmnist.setup(stage=None)
    a=fmnist.train_dataloader()
    example_data, example_target=next(iter(a))

    print(f"Example_Data \nDim : {example_data.dim()}, Shape: {example_data.shape}, dtype: {example_data.dtype}, device: {example_data.device} ")
    print(f"Example_target \nDim : {example_target.dim()}, Shape: {example_target.shape}, dtype: {example_target.dtype}, device: {example_target.device} ") 

    fmnist_label={0:'Top', 1:'Trouser', 2:'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

    pltsize=1
    plt.figure(figsize=(10*pltsize, pltsize))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis('off')
        plt.imshow(example_data[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
        plt.title(fmnist_label[example_target[i].item()])
    plt.show()
        
        