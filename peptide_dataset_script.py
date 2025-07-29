# Supress pytorch pickle load warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Torch
import torch

# Logging
from tqdm import tqdm

# Library imports
import gdiffusion as gd
import util
import util.chem as chem
import util.visualization as vis
import util.stats as gdstats

import h5py
import numpy as np

device = util.util.get_device()
print(f"device: {device}")

DIFFUSION_PATH = "../large_files/saved_models/diffusion/molecule-diffusion-v1.pt"
SELFIES_VAE_PATH = "../large_files/saved_models/selfies_vae/selfies-vae.ckpt"
PEPTIDE_VAE_PATH = "../large_files/saved_models/peptide_vae/peptide-vae.ckpt"
PEPTIDE_VAE_VOCAB_PATH = "../large_files/saved_models/peptide_vae/vocab.json"
LOGP_PREDICTOR_PATH = "s../large_files/aved_models/logp/model-logp"

RAW_DATA_PATH = "../large_files/data/raw_peptide/peptide_raw_10M.csv"
PEPTIDE_DATASET_PATH = "../large_files/data/peptide_dataset.h5"
RAW_DATA_PATH_4P5 = "../large_files/data/raw_peptide/peptide_raw_4p5.csv"

peptide_10m_len = 10274724
peptide_4p5_len = 4500001
total_len = peptide_10m_len + peptide_4p5_len - 2
peptide_latent_dim = 256

# Read from peptide latent data:
def read_peptide_dataset_raw(i: int, data_path="../large_files/data/peptide_dataset.h5"):
    with h5py.File(data_path, 'r') as f:
        return f['PEPTIDES'][i], f['EXTINCT'][i], f['DATA_SOURCE'][i], f['LATENTS'][i]
    
def read_peptide_dataset(i: int, data_path="../large_files/data/peptide_dataset.h5"):
    with h5py.File(data_path, 'r') as f:
        raw_peptide, raw_extinct, raw_datasource, raw_latent = f['PEPTIDES'][i], f['EXTINCT'][i], f['DATA_SOURCE'][i], f['LATENTS'][i]
        peptide = raw_peptide.decode('utf-8')
        extinct = bool(raw_extinct)
        datasource = 'peptide_10M' if raw_datasource == 0 else 'peptide_4.5M'
        latent = raw_latent

    return peptide, latent, extinct, datasource

# attatch latents to dataset (this will take a long long time)

vae = gd.load_vae_peptides(PEPTIDE_VAE_PATH, PEPTIDE_VAE_VOCAB_PATH)

def attatch_latents(start_idx: int = 576000, vae_batch_size=64):
    num_batches_per_block = 1000
    block_size = num_batches_per_block * vae_batch_size
    
    with h5py.File(PEPTIDE_DATASET_PATH, 'r+') as f:
        peptide_ds = f['PEPTIDES']
        latents_ds = f['LATENTS']
        
        block_num = start_idx // block_size
        block_start_index = block_num * block_size
        
        try:
            for block_idx in range(block_start_index, total_len, block_size):
                print(f"Block Number: {block_idx // block_size} --- Start Idx: {block_idx}")
                peptide_block = peptide_ds[block_idx:block_idx + block_size]
                
                # Fix: total should be number of batches, not number of items
                num_batches = (len(peptide_block) + vae_batch_size - 1) // vae_batch_size
                
                for i in tqdm(range(0, len(peptide_block), vae_batch_size), 
                             total=num_batches, desc='VAE Block'):
                    vae_batch = peptide_block[i:i + vae_batch_size]
                    vae_batch = [vae_ele.decode('utf-8') for vae_ele in vae_batch]
                    latents = gd.peptides_to_latent(vae_batch, vae=vae).cpu()
                    latents_ds[block_idx + i:block_idx + i + len(vae_batch)] = latents
                    
        except Exception as e:
            print(f"Encountered error: {e}")
            print(f"block_idx: {block_idx}")
            print(f"block_num: {block_idx // block_size}")
            print(f"i: {i}")
            print(f"len(peptide_block): {len(peptide_block)}")
            print(f"vae_batch_size: {vae_batch_size}")
            print(f"corrupted: {block_idx + i} to {block_idx + i + len(vae_batch)}")

vb = attatch_latents(start_idx=576000)

print("Finished")

