import torch
import numpy as np

from torch.utils.data import Dataset

class LatentDataset(Dataset):
    def __init__(self, latents):
        self.latents = torch.from_numpy(latents).float().unsqueeze(1)

    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx]
    
    def load(data_path="data/molecule_latents.npy"):
        data = np.load(data_path)
        print(f"Loaded data, shape={data.shape}")
        return LatentDataset(latents=data)
