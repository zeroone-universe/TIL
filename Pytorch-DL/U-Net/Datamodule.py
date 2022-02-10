import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import pytorch_lightning as pl


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


color_array = np.random.choice(range(256), 3*1000).reshape(-1, 3)
label_model = KMeans(n_clusters = 10)
label_model.fit(color_array)


class CityscapeDataset(Dataset):
    def __init__(self, image_dir, label_model):
        self.image_dir=image_dir

        self.paths_jpg=self.get_paths_jpg(self.image_dir)
        self.label_model=label_model

    def __len__(self) :
        return len(self.paths_jpg)

    def __getitem__(self, index):
        path_jpg=self.paths_jpg[index]
        sample_image = Image.open(path_jpg).convert("RGB")
        np_image=np.array(sample_image)
        cityscape, label= self.split_image(np_image)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        label_class = torch.Tensor(label_class).long()
        cityscape = self.transform(cityscape)
        return cityscape, label_class

    
    def split_image(self,image):
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label

    def get_paths_jpg(self,paths):
        jpg_paths = []
        if type(paths) == str:
            paths = [paths]
    
    
        for path in paths:
            for root, dirs, files in os.walk(path):
            
                jpg_paths += [os.path.join(root, file) for file in files if os.path.splitext(file)[-1] == '.jpg']

        jpg_paths.sort(key = lambda x: os.path.split(x)[-1])

        return jpg_paths

    def transform(self, image) :
        transform_ops = transforms.Compose([
      	    		transforms.ToTensor(),
                    transforms.Normalize(mean = (0.485, 0.56, 0.406), std = (0.229, 0.224, 0.225))
        ])
        return transform_ops(image)   


class CityscapeDatamodule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir= args.data_dir
        self.batch_size= args.batch_size


        self.dataset_train=CityscapeDataset(image_dir=f"{self.data_dir}/train")
        self.dataset_val=CityscapeDataset(image_dir=f"{self.data_dir}/val")

    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=0)
'''
    def predict_dataloader(self):
        optional
'''

if __name__=="__main__":
    
    test=CityscapeDatamodule(data_dir= "/media/youngwon/Neo/NeoChoi/TIL_Dataset/cityscapes_data",batch_size = 12)
    a=test.train_dataloader()
    img, label= next(iter(a))
    print(f"Example_Data \nDim : {img.dim()}, Shape: {img.shape}, dtype: {img.dtype}, device: {img.device} ")
    print(f"Example_target \nDim : {label.dim()}, Shape: {label.shape}, dtype: {label.dtype}, device: {label.device} ")