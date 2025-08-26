# Wrapper function around molformer VAE for SELFIEs and Peptides

import torch
from torch.nn.utils.rnn import pad_sequence

import json
from typing import List, Union

from gdiffusion.vae.molformers.models.BaseVAESwiGLURope import BaseVAE
from gdiffusion.vae.molformers.models.peptide_BaseTrainer import VAEModule as PeptideVAEModule
from gdiffusion.vae.molformers.models.BaseTrainer import VAEModule as SelfiesVAEModule

from util.chem.chem import selfies_to_smiles

# Supress pytorch pickle load warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class VAEWrapper:
    def __init__(self, vae_statedict_path : str, vocab_path : str, latent_dim: int, device : str):
        self.latent_dim = latent_dim
        self.device = device

    def _get_vocab(self, vocab_path : str):
        with open(vocab_path) as f:
            vocab = json.load(f)
        print(f"Loaded VAE Vocab from {vocab_path}")
        return vocab
    
    def _get_state_dict(self, vae_statedict_path: str):
        print(f"Getting State Dict...")
        state_dict = torch.load(vae_statedict_path, map_location=self.device)["state_dict"]
        
        print(f"Loading model from {vae_statedict_path}")

        # Remove 'model.' prefix if present (from nn.DataParallel)
        new_state_dict = {}
        for key in state_dict.keys():
            if key.startswith("model."):
                new_key = key[6:]  # remove the 'model.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]
        
        return new_state_dict
    
    def _forward(self, strings: Union[str, List[str]]):
        """Not Used. Convert Peptide/SELFIES string(s) to latent vector(s). Also returns the loss """
        if isinstance(strings, str):
            strings = [strings]

        
        # Convert Peptide/SELFIES to tokens using VAEModule's method
        tokens = []
        for string in strings:
            token_tensor = self.vae.string_to_tokens(string).to(self.device)
            tokens.append(token_tensor)
        tokens_batch = self._collate_fn(tokens)
        
        with torch.no_grad():
            out = self.vae.model(tokens_batch)
            z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])
            loss = out["loss"]
        
        # Reshape to match expected output format
        return z.reshape(-1, self.vae.model.n_acc * self.vae.model.d_bnk), loss


    def _collate_fn(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Not Used. Collate function for both SELFIES and Peptides"""
        return pad_sequence(batch, batch_first=True, padding_side=self.vocab['[pad]'])

        
    def _encode_helper(self, string: str) -> torch.Tensor:
        """Helper function. Takes in a Peptide/SELFIE string (singular) and returns its latent"""
        with torch.no_grad():
            tokens = self.string_to_tokens_fn(string).unsqueeze(0).to(self.device)
            out = self.vae.model(tokens)
            z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])

        return z.reshape(-1, self.vae.model.n_acc * self.vae.model.d_bnk)

    def _encode(self, strings: Union[str, List[str]]) -> torch.Tensor:
        """Takes in a Peptide/SELFIES string or list of Peptidee/SELFIES strings and returns a (B, D) tensor of their latent values"""
        if isinstance(strings, str):
            strings = [strings] # convert singleton into list

        return torch.cat([self._encode_helper(string) for string in strings], dim=0)
    
    def _decode(self, z: torch.Tensor, argmax : bool = True, max_len : int = 256) -> List[str]:
        """Takes in a (B, D) tensor and returns the decoded Peptide/SELFIES strings as a list of strings"""
        z = z.to(self.device)
        
        with torch.no_grad():

            # Use VAEModule's sample method to generate tokens
            tokens = self.vae.sample(
                z.view(-1, self.vae.model.n_acc * self.vae.model.d_bnk),
                argmax=argmax,
                max_len=max_len
            )

            strings_list = []
            for token_seq in tokens:
                string = self.tokens_to_string_fn(token_seq, drop_after_stop=True)
                strings_list.append(string)

            return strings_list
        
    def sample_random(self, batch_size: int, argmax: bool = True, max_len: int = 256) -> List[str]:
        """Sample a random batch from the Latent Space with VAE"""
        z_rand = torch.randn(size=(batch_size, self.latent_dim))
        return self._decode(z=z_rand, argmax=argmax, max_len=max_len)        

class MoleculeVAE(VAEWrapper):
    def __init__(
        self, 
        vae_statedict_path : str = "saved_models/selfies_vae/selfies-vae.ckpt",
        vocab_path : str = "saved_models/selfies_vae/vocab.json",
        device : str = 'cuda'
    ):
        print(f"\n Loading Molecule VAE:\n------------------------------------------------")
        latent_dim = 128 # molecule latent dim size
        
        super().__init__(vae_statedict_path, vocab_path, latent_dim, device)
        vocab = self._get_vocab(vocab_path=vocab_path)
        self.vocab = vocab
        state_dict = self._get_state_dict(vae_statedict_path=vae_statedict_path)
                
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
        ).to(self.device)

        model.load_state_dict(state_dict)
        model = model.to(self.device).eval()

        vae = SelfiesVAEModule(model).to(self.device).eval()
        self.vae = vae
        self.tokens_to_string_fn = self.vae.tokens_to_selfie
        self.string_to_tokens_fn = self.vae.selfie_to_tokens
        print("------------------------------------------------\n")
        

    def encode(self, selfies: Union[str, List[str]]) -> torch.Tensor:
        """Takes in a SELFIES string or list of SELFIES strings and returns a (B, D) tensor of their latent values"""
        return self._encode(selfies)
    
    def decode(self, z: torch.Tensor, argmax : bool = True, max_len : int = 256) -> List[str]:
        """Takes in a (B, D) tensor and returns the decoded SELFIES strings as a list of strings"""
        return self._decode(z, argmax=argmax, max_len=max_len)

    def decode_to_smiles(self, z: torch.Tensor, argmax: bool = True, max_len : int = 256) -> List[str]:
        return selfies_to_smiles(self.decode(z))
    
class PeptideVAE(VAEWrapper):
    def __init__(
        self, 
        vae_statedict_path : str = "saved_models/peptide_vae/peptide-vae.ckpt",
        vocab_path : str = "saved_models/peptide_vae/vocab.json",
        device : str = 'cuda'
    ):
        print(f"\n Loading Peptide VAE:\n------------------------------------------------")
        latent_dim = 256

        super().__init__(vae_statedict_path, vocab_path, latent_dim, device)
        vocab = self._get_vocab(vocab_path=vocab_path)
        self.vocab = vocab
        state_dict = self._get_state_dict(vae_statedict_path=vae_statedict_path)

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
        ).to(self.device)

        model.load_state_dict(state_dict)
        model = model.to(self.device).eval()

        vae = PeptideVAEModule(model).to(self.device).eval()
        self.vae = vae
        self.tokens_to_string_fn = self.vae.tokens_to_peptide
        self.string_to_tokens_fn = self.vae.peptide_to_tokens
        print("------------------------------------------------\n")


    def encode(self, peptides: Union[str, List[str]]) -> torch.Tensor:
        """Takes in a Peptide string or list of Peptide strings and returns a (B, D) tensor of their latent values"""
        return self._encode(peptides)
    
    def decode(self, z: torch.Tensor, argmax : bool = True, max_len : int = 256) -> List[str]:
        """Takes in a (B, D) tensor and returns the decoded Peptide strings as a list of strings"""
        return self._decode(z, argmax=argmax, max_len=max_len)
   