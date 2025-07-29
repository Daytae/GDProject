from pathlib import Path
from typing import List

import lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from peptide_molformers.datamodules.peptide_batch_sampler import SequenceLengthBatchSampler


class PeptideDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, vocab: dict[str, int], batch_size: int, num_workers: int, len_sample: bool = False) -> None:
        super().__init__()

        self.data_root = data_root
        self.vocab = vocab

        self.batch_size = batch_size
        self.num_workers = num_workers

        if '[start]' not in self.vocab:
            raise ValueError("Vocab must contain '[start]' token")
        if '[stop]' not in self.vocab:
            raise ValueError("Vocab must contain '[stop]' token")
        if '[pad]' not in self.vocab:
            raise ValueError("Vocab must contain '[pad]' token")
        
        self.len_sample = len_sample

    def train_dataloader(self) -> DataLoader:
        ds = PeptideDataset(self.data_root, 'train', self.vocab)
        if self.len_sample:
            return DataLoader(
                ds,
                num_workers=self.num_workers,
                collate_fn=ds.get_collate_fn(),
                batch_sampler=SequenceLengthBatchSampler(ds.peptides, bucket_size=5, batch_size=self.batch_size)
            )
        else:
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=ds.get_collate_fn()
            )

    def val_dataloader(self) -> DataLoader:
        ds = PeptideDataset(self.data_root, 'val', self.vocab)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ds.get_collate_fn()
        )

    def test_dataloader(self) -> DataLoader:
        ds = PeptideDataset(self.data_root, 'test', self.vocab)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ds.get_collate_fn()
        )

class PeptideDataset(Dataset):
    def __init__(self, data_root: str, split: str, vocab: dict[str, int]) -> None:
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.vocab = vocab

        # Try different file extensions
        path = Path(data_root) / f'{split}_peptide.txt'
        if not path.exists():
            path = Path(data_root) / f'{split}_peptide.csv'
        if not path.exists():
            path = Path(data_root) / f'{split}.txt'
        if not path.exists():
            path = Path(data_root) / f'{split}.csv'

        with open(path, 'r') as f:
            self.peptides = [l.strip() for l in f.readlines() if l.strip()]

    def __len__(self) -> int:
        return len(self.peptides)

    def __getitem__(self, index: int) -> torch.Tensor:
        peptide = f"[start]{self.peptides[index]}[stop]"
        
        # Tokenize peptide - each character is a token
        tokens = []
        i = 0
        while i < len(peptide):
            if peptide[i] == '[':
                # Find the end of the special token
                end = peptide.find(']', i)
                if end != -1:
                    token = peptide[i:end+1]
                    tokens.append(token)
                    i = end + 1
                else:
                    tokens.append(peptide[i])
                    i += 1
            else:
                tokens.append(peptide[i])
                i += 1
        
        tokens = torch.tensor([self.vocab[tok] for tok in tokens])
        return tokens

    def get_collate_fn(self):
        def collate(batch: List[torch.Tensor]) -> torch.Tensor:
            return pad_sequence(batch, batch_first=True, padding_value=self.vocab['[pad]'])
        return collate

# Simple tokenization for peptides
def peptide_tokenize(peptide: str) -> list[str]:
    """Tokenize peptide sequence into individual amino acids and special tokens"""
    tokens = []
    i = 0
    while i < len(peptide):
        if peptide[i] == '[':
            # Find the end of the special token
            end = peptide.find(']', i)
            if end != -1:
                token = peptide[i:end+1]
                tokens.append(token)
                i = end + 1
            else:
                tokens.append(peptide[i])
                i += 1
        else:
            tokens.append(peptide[i])
            i += 1
    return tokens