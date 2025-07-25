from torch.utils.data import Dataset, DataLoader
import h5py

class MoleculeDataset(Dataset):
    """Dataset for the Guacamole Molecules"""

    def __init__(self, file_loc:str ="data/molecule_dataset.h5", transform=None):
        """
        Arguments:
            file_loc (string): Path to the molecule dataset
            transform: transform to be applied on a sample
        """
        
        # keep file open
        self.file = h5py.File(file_loc, 'r')
        self.latent_dataset = self.file['LATENTS']
        self.smiles_dataset = self.file['SMILES']
        self.selfies_dataset = self.file['SELFIES']
        self._cached_len = len(self.smiles_dataset[:])
        self.transform = transform

    def __len__(self, use_cached=True):
        if use_cached:
            return self._cached_len
        else:
            return len(self.smiles_dataset[:])
    
    def __getitem__(self, idx):
        out = (self.smiles_dataset[idx], self.selfies_dataset[idx], self.latent_dataset[idx])
        if self.transform:
            out = self.transform(out)
        return out
    
    def get_smile(self, idx):
        return self.smiles_dataset[idx]
    
    def get_selfie(self, idx):
        return self.selfies_dataset[idx]
    
    def get_latent(self, idx):
        return self.latent_dataset[idx]
    
    