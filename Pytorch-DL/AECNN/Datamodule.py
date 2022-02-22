import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio as ta
import numpy as np

import pytorch_lightning as pl

from utils import *

class CEDataset(Dataset): 
  #데이터셋의 전처리를 해주는 부분
    def __init__(self, path_dir_dist, path_dir_orig, seg_len=2):
        self.path_dir_orig   = path_dir_orig  
        self.path_dir_dist   = path_dir_dist  
        self.seg_len = seg_len
        self.wavs={}
        self.filenames= []


        paths_wav_orig = get_wav_paths(self.path_dir_orig)
        paths_wav_dist= get_wav_paths(self.path_dir_dist)

        for path_wav_orig, path_wav_dist in zip(paths_wav_orig, paths_wav_dist):
            filename=get_filename(path_wav_orig)[0]
            wav_orig,_=ta.load(path_wav_orig)
            wav_dist,_=ta.load(path_wav_dist)
            self.wavs[filename]=(wav_orig, wav_dist)
            self.filenames.append(filename)

        self.filenames.sort()
        

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.filenames)


    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):

        filename = self.filenames[idx]
        (wav_orig, wav_dist) = self.wavs[filename]
        
        if self.seg_len>0:
            duration= int(self.seg_len * 16000)

            wav_orig= wav_orig.view(1,-1)
            wav_dist= wav_dist.view(1,-1)

            sig_len = wav_orig.shape[-1]

            t_start = np.random.randint(
                low = 0,
                high= np.max([1, sig_len- duration - 2]),
                size = 1
            )[0]
            t_end = t_start + duration

            wav_orig = wav_orig.repeat(1, t_end // sig_len + 1) [ ..., t_start : t_end]
            wav_dist = wav_dist.repeat(1, t_end// sig_len + 1) [ ..., t_start : t_end]

        return wav_dist, wav_orig #, filename




        


class CEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="/media/youngwon/Neo/NeoChoi/TIL_Dataset/AECNN_enhancement", batch_size=4, seg_len=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seg_len = seg_len

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        full_dataset = CEDataset(path_dir_orig = f"{self.data_dir}/target", path_dir_dist = f"{self.data_dir}/decoded", seg_len = self.seg_len)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [4000, len(full_dataset) - 4000])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size)
    

if __name__=="__main__":
    datamodule= CEDataModule()
    datamodule.setup()
    traindm=datamodule.train_dataloader()
    valdm=datamodule.val_dataloader()
    print(next(iter(traindm)))