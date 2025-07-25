from torch.utils.data import Dataset, DataLoader
import h5py
import torch

class LatentDataset(Dataset):
    """Dataset for a Latent Dataset"""

    def __init__(self, file_loc:str ="data/molecule_dataset.h5", latent_dim=None, latent_name='LATENTS', transform=None, load_in_memory=False):
        """
        Arguments:
            file_loc (string): Path to the dataset with the latents
            latent_dim: (int | None): the dimensionality of the latent space. None will have the dataset assume latent space dimensionality
            latent_name: (str): the name of the Latent dataset in the h5 dataset
            transform: transform to be applied on a sample. The samaple is automatically reshaped into a (B, DIM) tensor
            load_in_memory (bool): If True, loads all of the dataset into memory at once. Not recommended
        """
        
        # keep file open
        self.file = h5py.File(file_loc, 'r')
        self.latent_dataset = self.file[latent_name]

        # will load
        if load_in_memory:
            self.latent_dataset = self.file[latent_name][:]
            
        self._cached_len = len(self.latent_dataset[:])
        self.transform = transform

        if not latent_dim:
            # figure out latent dim
            if self._cached_len == 0:
                print("Warning: could not find latent dim dimension of empty dataset!")

            self.latent_dim = self.latent_dataset[0].shape[0]
        else:
            self.latent_dim = latent_dim

    def __len__(self, use_cached=True):
        if use_cached:
            return self._cached_len
        else:
            return len(self.latent_dataset[:])
    
    def __getitem__(self, idx):
        out = torch.tensor(self.latent_dataset[idx], dtype=torch.float32)
        out = out.reshape(-1, self.latent_dim)

        if self.transform:
            out = self.transform(out)
        return out
    