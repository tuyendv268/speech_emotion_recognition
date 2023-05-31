from torch.utils.data import Dataset, DataLoader
from src.utils import spec_augment
from src.feature_extractor import Feature_Extractor
import torch
import numpy as np
import os

class SER_Dataset(Dataset):
    def __init__(self, df, config, mode="train"):
        self.feature_extractor = Feature_Extractor(config=config)
        self.df = df
        self.mode = mode
        self.config = config
        self.n_label = len(config["label"].keys())
        self.max_duration = config["audio"]["max_duration"]
        
    def __len__(self):
        return self.df.shape[0]
    
    def smooth_labels(self, labels, factor=0.1):
        labels = labels.astype(np.float32)
        labels *= (1 - factor)
        labels += (factor / labels.shape[0])
        return labels
    
    def __getitem__(self, index):
        _path = self.df["path"][index]
        _label = self.df["label"][index]
        
        _label = self.config["label"][_label]
        _temp = np.zeros(self.n_label)
        _temp[_label] = 1 
        _label = _temp
        
        _mel = self.feature_extractor.extract_mel_spectrogram(_path, self.max_duration)
        
        if self.mode == "train":
            _mel = spec_augment(_mel)
            _label = self.smooth_labels(_label)
                    
        sample = {
            "input":_mel,
            "label":_label,
        }
        
        return sample
    
    def collate_fn(self, batch):
        lengths = [sample["input"].shape[1] for sample in batch]
        
        max_length = max(lengths)
        
        mels, masks, labels = [], [], []
        for sample in batch:
            mel = sample["input"]
            label = sample["label"]
            
            mask = [1] * mel.shape[1] + [0]*(max_length-mel.shape[1])
            paded_mel = np.pad(mel, ((0, 0), (0, max_length-mel.shape[1])), mode='constant', constant_values=0)
            
            mels.append(torch.tensor(paded_mel, dtype=torch.float32))
            masks.append(torch.tensor(mask, dtype=torch.bool))
            labels.append(torch.tensor(label, dtype=torch.float32))
            
        samples = {
            "inputs": torch.stack(mels, dim=0),
            "masks":torch.stack(masks, dim=0),
            "labels":torch.stack(labels, dim=0)
        }
        
        return samples