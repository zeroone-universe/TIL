import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

class CustomDataset(Dataset): 
  #데이터셋의 전처리를 해주는 부분
  def __init__(self):

  # 총 데이터의 개수를 리턴
  def __len__(self):
      return

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
      return


class PLLightningDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        
    
    def prepare_data(self):
        
    
    def setup(self, stage):
        
    
    def train_dataloader(self):
        return 
    
    def val_dataloader(self):
        return 
    
    def test_dataloader(self):
        return 
'''
    def predict_dataloader(self):
        optional
'''
    def tearsdown