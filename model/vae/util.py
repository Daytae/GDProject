import json
from typing import List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as pl
import selfies as sf

from model.vae.molformers.models.BaseTrainer import VAEModule
from model.vae.molformers.models.BaseVAESwiGLURope import BaseVAE
from typing import List, Union

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------------------------------------------------

#
# Decoders and Encoders:
#
def selfies_to_smiles(selfies):
    return [sf.decoder(selfie) for selfie in selfies]

# Rarely used
def smiles_to_selfies(smiles, strict=True, null_selfie=""):
    ''' Rarely Used. Only used for encoding training data into VAE representation'''
    
    selfies = []
    num_errors = 0
    
    for smile in smiles:
        try:
            selfie = sf.encoder(smile, strict=strict)

        except Exception as e:
            print(f"Encountered exception {e} while trying to convert {smile} to selfie representation")
            num_errors += 1
            selfie = null_selfie

        selfies.append(selfie)

    return selfies

def load_vae_selfies(path_to_vae_statedict="saved_models/selfies_vae/selfies-vae.ckpt", vocab_path="saved_models/selfies_vae/vocab.json"):
    """Load a VAE model for SELFIES representation"""

    with open(vocab_path) as f:
        vocab = json.load(f)
    
    model = BaseVAE(
        vocab,
        d_bnk=16,
        n_acc=8,
        
        d_dec=64,
        decoder_num_layers=3,
        decoder_dim_ff=256,
        
        d_enc=256,
        encoder_dim_ff=512,
        encoder_num_layers=3,
    ).to(device)
    
    state_dict = torch.load(path_to_vae_statedict, map_location=device)["state_dict"]
    
    print(f"loading model from {path_to_vae_statedict}")
    
    # Remove 'model.' prefix if present (from nn.DataParallel)
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("model."):
            new_key = key[6:]  # remove the 'model.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = state_dict[key]
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    vae = VAEModule(model).eval().to(device)
    return vae


def collate_selfies_fn(batch: List[torch.Tensor], vocab) -> torch.Tensor:
    """Collate function for SELFIES tokens"""
    return pad_sequence(batch, batch_first=True, padding_value=vocab['[pad]'])


def forward_selfies(selfies: Union[str, List[str]], vae):
    """Convert SELFIES string(s) to latent vector(s). Also returns the loss """
    if isinstance(selfies, str):
        selfies = [selfies]
    
    # Convert SELFIES to tokens using VAEModule's method
    tokens = []
    for s in selfies:
        token_tensor = vae.selfie_to_tokens(s).to(device)
        tokens.append(token_tensor)
    tokens_batch = collate_selfies_fn(tokens, vae.vocab)
    
    with torch.no_grad():
        out = vae.model(tokens_batch)
        z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])
        loss = out["loss"]
    
    # Reshape to match expected output format
    return z.reshape(-1, vae.model.n_acc * vae.model.d_bnk), loss


def latent_to_selfies_batch(z: torch.Tensor, vae, argmax=True, max_len=256):
    """Convert batch of latent vectors to SELFIES strings"""
    z = z.to(device)
    
    with torch.no_grad():
        # Use VAEModule's sample method to generate tokens
        tokens = vae.sample(
            z.view(-1, vae.model.n_acc * vae.model.d_bnk),
            argmax=argmax,
            max_len=max_len
        )
    
    # Convert tokens to SELFIES strings
    selfies_list = []
    for token_seq in tokens:
        selfie = vae.tokens_to_selfie(token_seq, drop_after_stop=True)
        selfies_list.append(selfie)
    
    return selfies_list


def latent_to_selfies(z: torch.Tensor, vae, argmax=True, max_len=256):
    """Convert latent vector(s) to SELFIES string(s)
    Wrapper around latent_to_selfies_batch for consistency
    """
    z = z.to(device)
    results = latent_to_selfies_batch(z, vae, argmax=argmax, max_len=max_len)
    return results

def latent_to_smiles(z: torch.Tensor, vae, argmax=True, max_len=256):
    """Convert latent vector(s) to SMILEs, which is what rdkit takes """
    selfies = latent_to_selfies(z=z, vae=vae, argmax=argmax, max_len=max_len)
    return selfies_to_smiles(selfies)

# Helper function to convert between single SELFIES and latent (without loss calculation)
def selfies_to_latent_helper(selfie: str, vae):
    """Convert a single SELFIES string to latent vector without calculating loss"""
    with torch.no_grad():
        tokens = vae.selfie_to_tokens(selfie).unsqueeze(0).to(device)
        out = vae.model(tokens)
        z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])
    
    return z.reshape(-1, vae.model.n_acc * vae.model.d_bnk)


def selfies_to_latent(selfies: Union[str, List[str]], vae):
    """Convert SELFIES string(s) to latent vector(s) without loss calculation"""
    if isinstance(selfies, str):
        return selfies_to_latent_helper(selfies, vae)
    elif isinstance(selfies, list):
        return torch.cat([selfies_to_latent_helper(s, vae) for s in selfies], dim=0)
    

def load_vae_peptides(path_to_vae_statedict, vocab_path="data/peptide_vocab.json"):
    """Load a VAE model for peptide representation"""
    
    # Load vocabulary
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    # Initialize model with same architecture as your training script
    model = BaseVAE(
        vocab,
        d_bnk=32,
        n_acc=8,
        
        d_dec=64,
        decoder_num_layers=4,
        decoder_dim_ff=256,
        
        d_enc=256,
        encoder_dim_ff=512,
        encoder_num_layers=4,
    )
    
    # Load state dict
    state_dict = torch.load(path_to_vae_statedict, map_location=device)["state_dict"]
    
    print(f"loading model from {path_to_vae_statedict}")
    
    # Remove 'model.' prefix if present (from nn.DataParallel)
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("model."):
            new_key = key[6:]  # remove the 'model.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = state_dict[key]
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    # Wrap in VAEModule
    vae = VAEModule(model).eval().to(device)
    
    return vae


def collate_peptides_fn(batch: List[torch.Tensor], vocab) -> torch.Tensor:
    """Collate function for peptide tokens"""
    return pad_sequence(batch, batch_first=True, padding_value=vocab['[pad]'])


def forward_peptides(peptides: Union[str, List[str]], vae):
    """Convert peptide string(s) to latent vector(s)
    Also returns the loss
    """
    # Ensure input is a list
    if isinstance(peptides, str):
        peptides = [peptides]
    
    # Convert peptides to tokens using VAEModule's method
    tokens = []
    for p in peptides:
        token_tensor = vae.peptide_to_tokens(p).to(device)
        tokens.append(token_tensor)
    
    # Collate tokens
    tokens_batch = collate_peptides_fn(tokens, vae.vocab)
    
    with torch.no_grad():
        out = vae.model(tokens_batch)
        z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])
        loss = out["loss"]
    
    # Reshape to match expected output format
    return z.reshape(-1, vae.model.n_acc * vae.model.d_bnk), loss


def latent_to_peptides_batch(z: torch.Tensor, vae, argmax=True, max_len=256):
    """Convert batch of latent vectors to peptide strings"""
    z = z.to(device)
    
    with torch.no_grad():
        # Use VAEModule's sample method to generate tokens
        tokens = vae.sample(
            z.view(-1, vae.model.n_acc * vae.model.d_bnk),
            argmax=argmax,
            max_len=max_len
        )
    
    # Convert tokens to peptide strings
    peptides_list = []
    for token_seq in tokens:
        peptide = vae.tokens_to_peptide(token_seq, drop_after_stop=True)
        peptides_list.append(peptide)
    
    return peptides_list


def latent_to_peptides(z: torch.Tensor, vae, argmax=True, max_len=256):
    """Convert latent vector(s) to peptide string(s)
    Wrapper around latent_to_peptides_batch for consistency
    """
    z = z.to(device)
    results = latent_to_peptides_batch(z, vae, argmax=argmax, max_len=max_len)
    return results


# Helper function to convert between single peptide and latent (without loss calculation)
def peptide_to_latent_helper(peptide: str, vae):
    """Convert a single peptide string to latent vector without calculating loss"""
    with torch.no_grad():
        tokens = vae.peptide_to_tokens(peptide).unsqueeze(0).to(device)
        out = vae.model(tokens)
        z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])
    
    return z.reshape(-1, vae.model.n_acc * vae.model.d_bnk)


def peptides_to_latent(peptides: Union[str, List[str]], vae):
    """Convert peptide string(s) to latent vector(s) without loss calculation"""
    if isinstance(peptides, str):
        return peptide_to_latent_helper(peptides, vae)
    elif isinstance(peptides, list):
        return torch.cat([peptide_to_latent_helper(p, vae) for p in peptides], dim=0)

