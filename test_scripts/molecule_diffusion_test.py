# Changes working directory to root to allow imports to work

import os

def change_directory_to_root(root_name: str = "GDProject"):
    """Changes the directory to the root of the project"""
    current_folder = os.getcwd().split('/')[-1]
    if current_folder != root_name:
        os.chdir('..')

    print(f"New Current Directory is {os.getcwd()}")

change_directory_to_root()


import time
SAVE_PATH = f"./results/run_mol_diff_test_{int(time.time()*1000)}"

# Supress pytorch pickle load warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

# Library imports
import gdiffusion as gd
import util
import util.chem as chem
import util.visualization as vis
import util.stats as gdstats

from gdiffusion.classifier.logp_predictor import LogPPredictor
# Paths for all of the models:
from util.model_paths import *

device = util.get_device(print_device=True)


# Load Diffusion
molecule_diffusion_model = gd.MoleculeDiffusionModel(unet_state_dict_path=MOLECULE_DIFFUSION_MODEL_PATH, device=device)
diffusion_ddim = gd.DDIMSampler(diffusion_model=molecule_diffusion_model, sampling_timesteps=50)
diffusion_ddpm =  gd.DDPMSampler(diffusion_model=molecule_diffusion_model)


# Test if diffusion works normally:
z_diffusion_cached = diffusion_ddpm.sample(batch_size=64)


# DDPM - Test Diffusion Can get all latent values to be 0.5

cond_fn_normal_dist = gd.get_cond_fn_normal_analytical(mean=-0.5, sigma=0.01)
z = diffusion_ddpm.sample(batch_size=16, cond_fn=cond_fn_normal_dist)

# Should be normal centered around 0.2
vis.display_latent(z[0], save_path=SAVE_PATH)


# DDIM - Test Diffusion Can get all latent values to be 0.5

cond_fn_normal_dist = gd.get_cond_fn_normal_analytical(mean=-0.5, sigma=0.01)
z = diffusion_ddim.sample(batch_size=16, cond_fn=cond_fn_normal_dist, guidance_scale=0.25)

# Should be normal centered around 0.2
vis.display_latent(z[0], save_path=SAVE_PATH)

# DDIM is broken and does not work


# Load VAE and classifier
vae = gd.MoleculeVAE()

import sys
sys.modules['__main__'].LogPPredictor = LogPPredictor
logp_predictor : LogPPredictor = torch.load(LOGP_PREDICTOR_PATH, weights_only=False).to(device)


# Diffusion Test #2 and #3
# The goal is to get the shapes to work (this is just a shape test)

logp_distribution = Normal(loc=100.0, scale=1.0)

# log probability function that gives the probability of the logps of z
# under the logp_distribution

def log_prob_fn_logp(z):
    # We cant backprop thru the evaluate_logp function, so we 
    # instead have this predictor trained on latent, logp pairs:
    pred_logp = logp_predictor(z)

    # get the log_probability under the normal distribution with mean=20.0
    log_prob = logp_distribution.log_prob(pred_logp).sum(dim=0)

    # the log probability must be a single value
    assert(log_prob.shape == (1, ))
    return log_prob


# Pass random data through to check shapes work
z_random = torch.randn(16, 1, 128, device=device)

cond_fn_logp = gd.get_cond_fn(
    log_prob_fn=log_prob_fn_logp, 
    latent_dim=128,
    guidance_strength=1.0, 
    clip_grad=True, 
    clip_grad_max=1.0,
)

grad_log_prob = cond_fn_logp(mean = z_random, t = 0)
assert(z_random.shape == grad_log_prob.shape)

# Diffusion Test #3
# Try and guide the diffusion to a high LogP molecule:
z = diffusion_ddpm.sample(batch_size=64, cond_fn=cond_fn_logp)
vis.display_logp_info(z, z_diffusion_cached=z_diffusion_cached, vae=vae, save_path=SAVE_PATH) # legacy function

print(f"Minimum VAE Value: {z.flatten().min():.2f}")
print(f"Maximum VAE Value: {z.flatten().max():.2f}")


# LogP Test to check model is working
# DDIM is basically cheating...

logp_distribution = Normal(loc=100.0, scale=1.0)
def log_prob_fn_logp(z):
    z = torch.clamp(z, -3.0, 3.0)
    pred_logp = logp_predictor(z)
    log_prob = logp_distribution.log_prob(pred_logp).sum(dim=0)
    assert(log_prob.shape == (1, ))
    return log_prob

cond_fn_logp = gd.get_cond_fn(
    log_prob_fn=log_prob_fn_logp, 
    latent_dim=128,
    clip_grad=False
)

z_diffusion_cached_ddim= diffusion_ddim.sample(batch_size=64)
z = diffusion_ddim.sample(batch_size=64, cond_fn=cond_fn_logp)
vis.display_logp_info(z, z_diffusion_cached=z_diffusion_cached_ddim, vae=vae, save_path=SAVE_PATH)


# Diffusion Test #2 and #3
# The goal is to get the shapes to work (this is just a shape test)

logp_distribution = Normal(loc=100.0, scale=1.0)

# log probability function that gives the probability of the logps of z
# under the logp_distribution

def log_prob_fn_logp(z):
    # We cant backprop thru the evaluate_logp function, so we 
    # instead have this predictor trained on latent, logp pairs:
    pred_logp = logp_predictor(z)

    # get the log_probability under the normal distribution with mean=20.0
    log_prob = logp_distribution.log_prob(pred_logp).sum(dim=0)

    # the log probability must be a single value
    assert(log_prob.shape == (1, ))
    return log_prob


# Pass random data through to check shapes work
z_random = torch.randn(16, 1, 128, device=device)

cond_fn_logp = gd.get_cond_fn(
    log_prob_fn=log_prob_fn_logp, 
    latent_dim=128,
    guidance_strength=1.0, 
    clip_grad=True, 
    clip_grad_max=1.0,
)

grad_log_prob = cond_fn_logp(mean = z_random, t = 0)
assert(z_random.shape == grad_log_prob.shape)

# Diffusion Test #3
# Try and guide the diffusion to a high LogP molecule:
z = diffusion_ddim.sample(batch_size=64, cond_fn=cond_fn_logp, guidance_scale=4.0)
vis.display_logp_info(z, z_diffusion_cached=diffusion_ddim.sample(batch_size=64), vae=vae, save_path=SAVE_PATH) # legacy function

print(f"Minimum VAE Value: {z.flatten().min():.2f}")
print(f"Maximum VAE Value: {z.flatten().max():.2f}")


