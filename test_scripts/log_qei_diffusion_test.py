# Changes working directory to root to allow imports to work

import os

def change_directory_to_root(root_name: str = "GDProject"):
    """Changes the directory to the root of the project"""
    current_folder = os.getcwd().split('/')[-1]
    if current_folder != root_name:
        os.chdir('..')

    print(f"New Current Directory is {os.getcwd()}")

change_directory_to_root()


# Supress pytorch pickle load warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Torch
import torch
from torch.utils.data import DataLoader, TensorDataset

# BoTorch

from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

# GPyTorch

import gpytorch
from gpytorch.mlls import PredictiveLogLikelihood

# Logging
import time
import json
from tqdm import tqdm

# Library imports
import gdiffusion as gd
import util
import util.chem as chem

# Paths for all of the models:
from util.model_paths import *

# Import guacamole tasks:
import util.chem.guacamole as guac

device = util.get_device(print_device=True)
vae = gd.MoleculeVAE()

SAVE_PATH = f"./results/run_ei_diff_test_{int(time.time()*1000)}"


# === Surrogate Model ===

def update_surr_model(model, mll, learning_rte, train_z, train_y, n_epochs):
    model = model.train()
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": learning_rte}], lr=learning_rte)
    train_bsz = min(len(train_y), 128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in tqdm(range(n_epochs), leave=False):
        for inputs, scores in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()

    return model

# === Validation Function ===

def validate_with_descriptor_gp(diffusion_model, batch_sizes=[64], surr_iters=[16], log_path="./log.json"):
    print("=== Conditional Sampling (GP Condition) ===")
    ddim_sampler = gd.DDIMSampler(diffusion_model=diffusion_model, sampling_timesteps=1000)

    data_batch_size = 1024
    LATENT_DATASET_LOCATION = "data/latents_pair_dataset_1"
    X_all_data = torch.load(LATENT_DATASET_LOCATION, weights_only=True)

    def obj_fun(z: torch.Tensor):
        selfies = vae.decode(z.float())
        smiles = chem.selfies_to_smiles(selfies)
        guac_score = guac.smiles_to_desired_scores(smiles, task_id="rano")
        return torch.tensor(guac_score).reshape(-1, 1)
    
    latent_dim = 128
    bounds = torch.tensor([[-3.0] * latent_dim, [3.0] * latent_dim], device=device, dtype=torch.float64)

    def get_batch(idx):
        start, end = idx*data_batch_size, (idx+1)*data_batch_size
        if start > len(X_all_data):
            return None, None

        Xs = X_all_data[start:end].reshape(-1, latent_dim).clone().double().to(device)
        Ys = obj_fun(Xs.float()).reshape(-1, 1).clone().double().to(device)
        return Xs, Ys

    batch = get_batch(idx=0) # [b,8,16]
    inducing_z = batch[0].cuda()
    device = inducing_z.device

    surrogate_model = gd.GPModelDKL(inducing_z, likelihood=gpytorch.likelihoods.GaussianLikelihood().cuda()).cuda()
    surrogate_mll = PredictiveLogLikelihood(surrogate_model.likelihood, surrogate_model, num_data=data_batch_size)
    surrogate_model.eval()

    max_score = float('-inf')
    best_z = None

    from collections import defaultdict
    summary = defaultdict(dict)
    idx = 0
    while get_batch(idx) is not None:
        batch_z, batch_y = get_batch(idx)
        flat_batch_y = batch_y.squeeze(1)
        surrogate_model = update_surr_model(surrogate_model, surrogate_mll, 0.002, batch_z, flat_batch_y, 100)

        batch_max_score, batch_max_idx = flat_batch_y.max(dim=0)
        if batch_max_score.item() > max_score:
            max_score = batch_max_score.item()
            best_z = batch_z[batch_max_idx].detach().clone()

        if idx+1 in surr_iters:
            best_f = torch.tensor(max_score, device=device, dtype=torch.float32)

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
            log_qEI = qLogExpectedImprovement(model=surrogate_model.cuda(), best_f=best_f, sampler=sampler)
            qEI = qExpectedImprovement(model=surrogate_model.cuda(), best_f=best_f, sampler=sampler)
            torch.cuda.empty_cache()

            def log_prob_fn_ei(x, eps=1e-8, tau=None):
                vals = qEI(x).clamp_min(eps)
                if tau is None:
                    tau = torch.quantile(vals.detach(), 0.5).clamp_min(eps)
                squashed = vals / (vals + tau)
                return torch.log(squashed + eps)

            cond_fn_ei = gd.get_cond_fn(
                log_prob_fn=log_prob_fn_ei,
                clip_grad=False,
                latent_dim=latent_dim,
            )
            
            for batch_size in batch_sizes:
                print(f"processing (iter: {idx+1}, bsz: {batch_size})")
                curr_summary = {}

                num_restarts = 0
                log_qei_score = "N/A"
                while log_qei_score == "N/A" and num_restarts < 5:
                    try:
                        latents = ddim_sampler.sample(batch_size=batch_size, cond_fn=None, guidance_scale=1.0)
                        log_qei_score = log_qEI(latents.reshape(-1, latent_dim)).detach().cpu().item()
                    except Exception as e:
                        num_restarts += 1
                curr_summary['no cond'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                print(f"log qei no cond: {log_qei_score}")
                torch.cuda.empty_cache()

                num_restarts = 0
                log_qei_score = "N/A"
                while log_qei_score == "N/A" and num_restarts < 3:
                    try:
                        latents = ddim_sampler.sample(batch_size=batch_size, cond_fn=cond_fn_ei, guidance_scale=25.0)
                        log_qei_score = log_qEI(latents.reshape(-1, latent_dim)).detach().cpu().item()
                    except Exception as e:
                        num_restarts += 1
                curr_summary['ddim'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                print(f"log qei ddim: {log_qei_score}")
                torch.cuda.empty_cache()

                num_restarts = 0
                log_qei_score = "N/A"
                while log_qei_score == "N/A" and num_restarts < 3:
                    try:
                        latents, _ = optimize_acqf(log_qEI, bounds=bounds, q=batch_size, num_restarts=10, raw_samples=1024)
                        log_qei_score = log_qEI(latents.reshape(-1, latent_dim)).detach().cpu().item()
                    except Exception as e:
                        num_restarts += 1
                curr_summary['optimize acqf'] = {'log qei score': log_qei_score, 'num restarts': num_restarts}
                print(f"log qei optimize acqf: {log_qei_score}")
                torch.cuda.empty_cache()

                summary[f"(iter: {idx+1}, bsz: {batch_size})"] = curr_summary
                with open(log_path, 'w') as file:
                    json.dump(summary, file, indent=2)
        
        if idx > max(surr_iters):
            break
    
    with open(log_path, 'w') as file:
        json.dump(summary, file, indent=2)
    print(summary)

# === Entry point ===

def main():
    molecule_diffusion_model = gd.MoleculeDiffusionModel(unet_state_dict_path=MOLECULE_DIFFUSION_MODEL_PATH, device=device)

    os.makedirs(SAVE_PATH, exist_ok=True)
    log_path = os.path.join(SAVE_PATH, 'log.json')
    validate_with_descriptor_gp(diffusion=molecule_diffusion_model, batch_sizes=[4, 8, 16, 32, 64, 128, 256], surr_iters = [1, 4, 16, 64, 256], log_path=log_path)

if __name__ == "__main__":
    main()