from torch.utils.data import Dataset, DataLoader
import h5py

class PeptideDataset(Dataset):
    """Dataset for the Peptides"""

    def __init__(self, file_loc:str ="data/peptide_dataset.h5", transform=None):
        """
        Arguments:
            file_loc (string): Path to the peptide dataset
            transform: transform to be applied on a sample
        """
        
        # keep file open
        self.file = h5py.File(file_loc, 'r')
        self.latent_dataset = self.file['LATENTS']
        self.peptide_dataset = self.file['PEPTIDES']
        self.extinct_dataset = self.file['EXTINCT']
        self.datasource_dataset = self.file['DATA_SOURCE']
        self._cached_len = len(self.peptide_dataset[:])
        self.transform = transform

    def __len__(self, use_cached=True):
        if use_cached:
            return self._cached_len
        else:
            return len(self.peptide_dataset[:])
    
    def __getitem__(self, idx):
        peptide = self.get_peptide(idx)
        latent = self.get_latent(idx)
        extinct = self.get_extinct(idx)
        datasource = self.get_datasource(idx)

        out = (peptide, latent, extinct, datasource)
        
        if self.transform:
            out = self.transform(out)
        return out
      

    def transform_peptide(peptide):
        ''' Converts peptide from binary format to normal string format'''
        return [ptd.decode('utf-8') for ptd in peptide] if isinstance(peptide, list) else peptide.decode('utf-8')
    
    def transform_datasource(datasource):
        ''' Turns the 0/1 labeling into string labels'''
        label_fn = lambda ds: 'peptide_10M' if datasource == 0 else 'peptide_4.5M'
        return [label_fn(ds) for ds in datasource] if isinstance(datasource, list) else label_fn(datasource)

    def get_peptide(self, idx, raw=True):
        peptide = self.peptide_dataset[idx]

        if not raw:
            peptide = self.transform_peptide(peptide)
        return peptide
    
    def get_latent(self, idx):
        return self.latent_dataset[idx]
    
    def get_extinct(self, idx):
        return self.extinct_dataset[idx]
    
    def get_datasource(self, idx, raw=True):
        datasource = self.datasource_dataset[idx]

        if not raw:
            datasource = self.transform_datasource(datasource)
        return datasource
    
