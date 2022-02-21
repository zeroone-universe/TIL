import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio as ta

import pytorch_lightning as pl

from utils import *

class CodecEnhancement(Dataset): 
  #데이터셋의 전처리를 해주는 부분
    def __init__(self, path_dir_orig, path_dir_dist, seg_len):
        self.path_dir_orig   = path_dir_orig  
        self.path_dir_dist   = path_dir_dist  
        self.seg_len = seg_len

        paths_wav_orig = get_paths_wav(self.path_dir_orig)
        paths_wav_dist= get_paths_wav(self.path_dir_dist)

        for path_wav_orig, path_wav_dist in zip(paths_wav_orig, paths_wav_dist):
            filename=get_filename(path_wav_orig)[0]
            wav_orig,_=ta.load(path_wav_orig)
            wav_dist,_=ta.load(path_wav_dist)
            self.wavs[filename]=(wav_orig, wav_dist)
            self.filenames.append(filename)
        

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.filenames)



    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):

        filename = self.filenames[idx]
        (wave_orig, wav_dist) = self.wavs[filename]
        
        if self.seg_len>0:
            duration= int(self.seg_len * 16000)

            wav_orig= wav_orig.view(1,-1)
            wav_dist= wav_dist.view(1,-1)

            sig_len = wav_orig[-1]

            t_start = np.random.randint(
                low = 0,
                high= np.max([1, sig_len- duration - 2]),
                size = 1
            )[0]
            t_end = t_start + duration

            wav_orig = wav_orig.repeat(1, t_end // sig_len + 1) [ ..., t_start : t_end]
            wav_dist = wav_dist.repeat(1, t_end// sig_len + 1) [ ..., t_start : t_end]

        return wav_orig, wav_dist, filename



        


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