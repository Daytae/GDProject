from typing import Literal, NamedTuple

from gdiffusion.diffusion.unet1d import Unet1D
from gdiffusion.diffusion.beta_scheduler import BetaSchedule
from gdiffusion.diffusion.util import *

from functools import partial
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast

from einops import reduce
from tqdm.auto import tqdm

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
DiffusionObjective = Literal["pred_noise", "pred_v", "pred_x0"]

class LatentDiffusionModel(nn.Module):
    '''
        Diffusion model for 1D diffusion.
        Inputs:
            model (Unet1D): The transformer that predicts v/noise/x0. Must accept input
                in shape of (B, 1, D).

            beta_schedule (BetaSchedule):
                One of the BetaSchedule's defined
            
            objective (str): One of pred_v, pred_noise, pred_x0. pred_v is recommended

            latent_dim (int): The latent dimension of the latent diffusion model.

            num_timesteps (int): The number of timesteps during training

            device: device

            clip_denoised: If the model should clip the denoised images during sampling / training
            
    '''
    
    def __init__(self, 
        model: Unet1D, 
        beta_schedule: BetaSchedule,
        objective: DiffusionObjective = "pred_v",
        latent_dim: int = 128,
        num_timesteps: int = 1000,
        device=None,
        clip_denoised: bool = False,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
    ):

        # being a subclass of nn.Module allows us to register buffers
        super().__init__()

        self.device = device if device is not None else self._get_device()

        self.model = model 
        self.beta_schedule = beta_schedule
        self.objective = objective
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        
        self.clip_denoised = clip_denoised
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        assert self.clip_min < self.clip_max
        
        self.register_buffers()
      

    def register_buffers(self):
        betas = self.beta_schedule.get_betas(self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        loss_weight = self._get_loss_weight(snr)
        register_buffer('loss_weight', loss_weight)

        self.to(self.device)

    def _get_loss_weight(self, snr):
        snr = snr.clone()
        match self.objective:
            case "pred_v":
                return snr / snr
            case "pred_noise":
                return snr
            case "pred_x0":
                return snr / (snr + 1)
            case _:  # default case
                raise ValueError(f"Unknown objective: {self.objective}")
            
    def _get_device(self):
            return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t):
        clip_denoised, clip_min, clip_max = self.clip_denoised, self.clip_min, self.clip_max

        model_output = self.model(x, t)
        maybe_clip = partial(torch.clamp, min = clip_min, max = clip_max) if clip_denoised else identity

        match self.objective:
            case "pred_v":
                v = model_output
                x_start = self.predict_start_from_v(x, t, v)
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(x, t, x_start)

            case "pred_noise":
                pred_noise = model_output
                x_start = self.predict_start_from_noise(x, t, pred_noise)
                x_start = maybe_clip(x_start)

            case "pred_x0":
                x_start = model_output
                x_start = maybe_clip(x_start)
                pred_noise = self.predict_noise_from_start(x, t, x_start)

            case _:  # default case
                raise ValueError(f"Unknown objective: {self.objective}")
                
        return ModelPrediction(pred_noise, x_start)
    
    def p_mean_variance(self, x, t):
        preds = self.model_predictions(x, t)
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
     
    def condition_mean(self, cond_fn, mean, variance, t):
        """
            Compute the mean for the previous step, given a function cond_fn that
            computes the gradient of a conditional log probability with respect to
            x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
            condition on y.
            This uses the conditioning strategy from Sohl-Dickstein et al. (2015).

            # this fixes a bug in the official OpenAI implementation:
            # https://github.com/openai/guided-diffusion/issues/51 (see point 1)
            # use the predicted mean for the previous timestep to compute gradient
        """
        
        gradient = cond_fn(mean, t)
        new_mean = (mean.float() + variance * gradient.float())

        # print("gradient: ",(gradient.float()))
        # print("gradient-mean: ",(variance * gradient.float()).mean())
        return new_mean

    @torch.no_grad()
    def p_sample(self, x, t: int, batch_size: int, cond_fn=None):
        batched_times = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times)

        if exists(cond_fn):
            model_mean = self.condition_mean(cond_fn=cond_fn, mean=model_mean, variance=variance, t=batched_times)
        
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_x, x_start

    @torch.no_grad()
    def sample(self, batch_size=16, cond_fn=None):
        shape, device = (batch_size, 1, self.latent_dim), self.device

        x = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'DDPM Sampling loop time step', total=self.num_timesteps):
            x, x_start = self.p_sample(x, t, batch_size=batch_size, cond_fn=cond_fn)

        return x

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def p_losses(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t=t, noise = noise)

        model_out = self.model(x, t)
        target = self.predict_v(x_start, t, noise) #v

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, x, *args, **kwargs):
        batch_size = x.shape[0]

        # dimension check
        dim = x.shape[2]
        assert dim == self.latent_dim, f'seq_length must be {self.latent_dim}'

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

        return self.p_losses(x, t, *args, **kwargs)
