import random
import numpy as np
import librosa
import pandas as pd
import os

def load_data(path):
    df = pd.read_csv(path, names=["id", "label"], sep="|", dtype={"id":str})
    
    return df

def prepare_data(metadata_path, wav_path):
    metadata = load_data(metadata_path)
    metadata["path"] = metadata.id.apply(lambda x: os.path.join(wav_path, x+".wav"))
    
    return metadata
    
def spec_augment(spec: np.ndarray, num_mask=1, 
                 freq_masking_max_percentage=0.1, time_masking_max_percentage=0.1):
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec
