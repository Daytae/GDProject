# Supress pytorch pickle load warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Torch
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

# Logging
from tqdm import tqdm

# Library imports
import gdiffusion as gd
import util
import util.chem as chem
import util.visualization as vis
import util.stats as gdstats

import datasets as ds

import h5py
import wandb
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Literal

from torch.optim import Adam
from ema_pytorch import EMA
from gdiffusion.classifier.extinct_predictor import EsmClassificationHead

device = util.util.get_device()
print(f"device: {device}")


DIFFUSION_PATH = "saved_models/diffusion/molecule-diffusion-v1.pt"
SELFIES_VAE_PATH = "saved_models/selfies_vae/selfies-vae.ckpt"
SELFIES_VOCAB_PATH = "saved_models/selfies_vae/vocab.json"

PEPTIDE_VAE_PATH = "saved_models/peptide_vae/peptide-vae.ckpt"
PEPTIDE_VOCAB_PATH = "saved_models/peptide_vae/vocab.json"

LOGP_PREDICTOR_PATH = "saved_models/logp/model-logp"
EXTINCT_PREDICTOR_PATH = "saved_models/extinct_model8417"
PEPTIDE_DATASET_PATH = "data/peptide_dataset.h5"
PEPTIDE_DATASET_LEN = 14774723 # speed up loading

EVAL_EVERY = 100_000
SAVE_EVERY = 100_000
NUM_CYCLES = 10_000_000

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class DiffusionTrainer:
    def __init__(
        self,
        diffusion_model : gd.LatentDiffusionModel, # diffusion model to train
        dataloader: DataLoader, # latent dataset to train on
        eval_fn, # function to run when we evaluate model
        eval_every = 10, # we should evaluate the model with eval_fn every X examples

        train_num_examples = 100, # number of examples to train on
            
        save_every = 10, # we should save model every X examples
        save_model_folder = "train/", # where to save the model

        train_lr=1e-4,
        adam_betas = (0.9, 0.99),
        ema_update_every = 10,
        ema_decay = 0.995,
        max_grad_norm=1.0,
        device = None
    ): 
        self.device = util.util.get_device(device=device)
        self.model = diffusion_model
        self.dataloader = gd.diffusion.util.cycle(dataloader)

        # evaluation
        self.eval_fn = eval_fn
        self.eval_every = eval_every
        
        self.train_batch_size = dataloader.batch_size
        self.train_num_examples = train_num_examples
        
        self.save_every = save_every
        self.save_model_folder = Path(save_model_folder)
        self.save_model_folder.mkdir(exist_ok=True)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.max_grad_norm = max_grad_norm
        self.ema.to(self.device)

        # progress bars
        self.step = 0
        self.dataset_len = len(dataloader.dataset) # hopefully is cached lola
        self.log_every = -1 # no logging 
        self.is_logging = self.log_every > 0

        print(f"Loaded {diffusion_model.latent_dim}-dimension latent diffusion model, optimizing {diffusion_model.objective}")
        print(f"Dataset has {self.dataset_len} elements, training will proceed on {self.train_num_examples} of them")
        print(f"Trainer will save every {self.save_every} to {self.save_model_folder} and evaluate every {self.eval_every}")
        print(f"Call .init_wandb() to initialize wandb logging")
        
    def init_wandb(self, 
        log_every: int, 
        name: str = None, 
        log_type: Literal['gradients', 'parameters', 'all'] | None = None,
        project="Guided Diffusion Project v2",
        wandb_dir="train"
    ):
        
        ''' Not called in __init__(), must be called manually'''
        self.log_every = log_every
        self.is_logging = True

        wandb.init(
            project=project,
            dir=wandb_dir,
            name=name
        )

        if log_type:
            wandb.watch(self.model, log=log_type)

    def _get_model_name(self, milestone):
        return str(self.save_model_folder / f'peptide_model_v1-{milestone}.pt')
    
    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'version': "3.0.0"
        }

        torch.save(data, self._get_model_name(milestone))

    def load(self, milestone, model_name=None):
        model_name = model_name if model_name is not None else self._get_model_name(milestone)
        print(f"Loading to device={self.device}...")
        data = torch.load(self._get_model_name(milestone), map_location=self.device, weights_only=False)

        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

    def _train_model_step(self):
        self.model.train()
        self.ema.ema_model.train()

        self.opt.zero_grad()
        data = next(self.dataloader).to(self.device)
        
        loss = self.model(data)

        loss.backward()
        loss_item = loss.detach().item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        self.opt.step()
        return loss_item

    def _eval_model(self):
        print("Attempting evaluation routine:")
        try:
            self.ema.ema_model.eval()
            self.model.eval()
            self.eval_fn(self.model)

        except Exception as e:
            print("Encounterred error in sampling routine, aborting")
            print(e)

    def _save_model(self):
        self.ema.ema_model.eval()
        
        with torch.no_grad():
            milestone = self.step // self.save_every
            print(f"Saving {self._get_model_name(milestone=milestone)} to {self.save_model_folder}")
            self.save(milestone=milestone)

    def _log_model(self, loss, pbar):
        pbar.set_description(f'loss: {loss:.4f}')
        log_dict = {
            "train/loss" : loss,
            "train/step": self.step,
            "train/epoch": self.step // self.dataset_len,
            "train/learning_rate": self.opt.param_groups[0]['lr'],
        }
        wandb.log(log_dict)

    def train(self):
        if not self.is_logging:
            print("Warning: wandb logging is not enabled, this may not be intentional")

        with tqdm(initial=self.step, total=self.train_num_examples) as pbar:
            while self.step < self.train_num_examples:
                if self.step % self.save_every == 0:
                    # self._save_model()
                    pass

                if self.step % self.eval_every == 0:
                    # self._eval_model()
                    pass

                # train model for 1 step
                loss = self._train_model_step()

                if self.step % self.log_every == 0 and self.is_logging:
                    self._log_model(loss, pbar)

                self.ema.update()
                self.step += 1
                pbar.update(1)
                    
        print("Training complete!")

peptide_latent_dataset = ds.LatentDataset(file_loc=PEPTIDE_DATASET_PATH, latent_dim=256, dataset_len=PEPTIDE_DATASET_LEN)
peptide_dataloader = DataLoader(
    dataset=peptide_latent_dataset, 
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    num_workers=0
)

diffusion_model = gd.create_diffusion_model(
    unet_dim=256, 
    diffusion_latent_dim=256, 
    objective='pred_v', 
    beta_schedule=gd.BetaScheduleSigmoid, 
    clip_denoised=False, 
    clip_min=-3.0,  
    clip_max=3.0, 
    model_path=None, 
    device=device
).to(device)

extinct_predictor : EsmClassificationHead = torch.load(EXTINCT_PREDICTOR_PATH).to(device)
extinct_predictor.eval()

def evaluate_model(diffusion_model: gd.LatentDiffusionModel, vae=None, is_logging=True):
    print("Evaluating Model...")
    batch_size = 16

    # sample from diffusion and VAE prior
    print("Doing statistical sample")
    z1 = diffusion_model.sample(batch_size=batch_size)
    # z2 = diffusion_model.sample(batch_size=batch_size)
    rand1 = torch.randn(size=(batch_size, 256))
    rand2 = torch.randn(size=(batch_size, 256))

    _, random_control_p = gdstats.is_different_from_other(z=rand1, z_other=rand2)
    _, diffusion_versus_random_p = gdstats.is_different_from_other(z=z1, z_other=rand1)

    # TODO: Convert z1 and compare it to a random peptide and see if there is similarity

    # TODO: Diffuse with extinct and see if extinct % goes up

    def log_prob_fn_extinct(z):
        batch_size, latent_dim = z.shape
        logits = extinct_predictor(z)
        log_prob_sum = F.log_softmax(input=logits, dim=-1).sum(dim=0)
        log_prob_sum[0] *= -1
        log_prob = log_prob_sum.sum(dim=0)
        return log_prob

    cond_fn_extinct = gd.get_cond_fn(
        log_prob_fn=log_prob_fn_extinct, 
        guidance_strength=1.0, 
        clip_grad=True, 
        clip_grad_max=1.0,
        latent_dim=256
    )

    print("Doing guided sample")
    z_guided = diffusion_model.sample(batch_size=batch_size, cond_fn=cond_fn_extinct)
    z_guided = z_guided.reshape(-1, 256)
    log_probs = F.log_softmax(z_guided, dim=-1)


    eval_dict = {
        "eval/random_control_p" : random_control_p,
        "eval/diffusion_versus_random_p" : diffusion_versus_random_p,
        "eval/guided_diffusion_log_prob_extinct" : log_probs
    }

    wandb.log(eval_dict)
    
trainer = DiffusionTrainer(
    diffusion_model = diffusion_model,
    dataloader = peptide_dataloader,
    eval_fn = evaluate_model,
    eval_every = EVAL_EVERY,
    train_num_examples = NUM_CYCLES,
    save_every = SAVE_EVERY,
    save_model_folder= "train/"
)

# trainer.init_wandb(log_every=1, name='Peptide Diffusion Train Run (Attempt #1)', project='Guided Diffusion Project v2')

try:
    trainer.train()
except Exception as e:
    print(f"Training error encounterd, saving model")
    print(e)

    trainer.ema.ema_model.eval()
    emergency_save_name = "model_emergency_save_123"
    data = {
        'step': trainer.step,
        'model': trainer.model.state_dict(),
        'opt': trainer.opt.state_dict(),
        'ema': trainer.ema.state_dict(),
        'version': "3.0.0"
    }

    torch.save(data, emergency_save_name)