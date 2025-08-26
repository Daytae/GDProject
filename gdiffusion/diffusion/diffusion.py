
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp import autocast
from functools import partial
from einops import reduce

from tqdm import tqdm

from typing import Tuple
from collections import namedtuple
from collections.abc import Callable

from gdiffusion.diffusion.beta_scheduler import BetaSchedule, BetaScheduleSigmoid
from gdiffusion.diffusion.util import *


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

    
class LatentDiffusionModel:
    def __init__(
        self,
        model: nn.Module, # Unet1D usually
        latent_dim: int,
        num_timesteps: int = 1000,
        clip_denoised: bool = False,
        clip_bounds : Tuple[float, float] = (-1.0, 1.0),
        beta_schedule : BetaSchedule = BetaScheduleSigmoid, # using sigmoid schedule by default
        objective= "pred_v", # only pred_v is supported for now
        device="cuda",

    ) -> None:
        self.model = model
        self.dim = latent_dim
        self.latent_dim = latent_dim

        self.num_timesteps = num_timesteps
        self.clip_denoised = clip_denoised
        self.clip_bounds = clip_bounds
        self.beta_schedule = beta_schedule
        self.objective = objective
        self.device = device

        clip_low, clip_high = self.clip_bounds
        assert(clip_low <= clip_high)

    def load_unet(self, state_dict_path: str):
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location=self.device)

            # Load only UNet
            unet_state_dict = {}
            for key in state_dict.keys():
                if key.startswith("model."):
                    new_key = key[6:]  # remove the 'model.' prefix
                    unet_state_dict[new_key] = state_dict[key]
            
            self.model.load_state_dict(unet_state_dict)

            print(f"[Molecule Diffusion]: UNet Successfully loaded")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"- Total parameters: {total_params:,}")
            print(f"- Trainable parameters: {trainable_params:,}")
            print(f"- Model size: {total_params * 4 / (1024**2):.1f} MB")
    
class DiffusionSampler(nn.Module):
    diffusion_model : LatentDiffusionModel
    model: nn.Module
    dim: int
    num_timesteps: int
    device: torch.device
    betas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    log_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor

    def __init__(
        self,
        diffusion_model : LatentDiffusionModel,
    ) -> None:
        super().__init__()
        self.diffusion_model = diffusion_model

        self.model = diffusion_model.model
        self.dim = diffusion_model.dim
        self.num_timesteps = diffusion_model.num_timesteps
        self.device = diffusion_model.device
        self.clip_denoised = diffusion_model.clip_denoised
        self.clip_min, self.clip_max = diffusion_model.clip_bounds


        # Using Sigmoid
        betas = diffusion_model.beta_schedule.get_betas(num_timesteps=self.num_timesteps)  # type: ignore
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        def register_buffer(name: str, val: torch.Tensor) -> None:
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        #---------------------------------------------------------------------------------
        #  For DDPM
        # --------------------------------------------------------------------------------
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        snr_clone = snr.clone()
        loss_weight = snr_clone / snr_clone # pred_v
        register_buffer('loss_weight', loss_weight)
        self.to(self.device)

    def load(self, state_dict_path : str):
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location=self.device)
            self.load_state_dict(state_dict=state_dict)

            print(f"[Diffusion Sampler]: Sampler Successfully loaded")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"- Total parameters: {total_params:,}")
            print(f"- Trainable parameters: {trainable_params:,}")
            print(f"- Model size: {total_params * 4 / (1024**2):.1f} MB")
        else:
            print(f"[Diffusion]: Warning - No state dict was given")

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        a_bar_sqrt = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        one_minus_a_bar_sqrt = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return one_minus_a_bar_sqrt * x_t + a_bar_sqrt * v
   

class DDIMSampler(DiffusionSampler):
    sampling_timesteps: int | None
    ddim_sampling_eta: float

    def __init__(
        self,
        diffusion_model : LatentDiffusionModel,
        sampling_timesteps: int | None = None,
        ddim_sampling_eta: float = 0.0,
    ) -> None:
        super().__init__(diffusion_model=diffusion_model)
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        guidance_scale: float = 1.0,
        cond_fn: Callable | None = None,
        reshape_output = True
    ) -> torch.Tensor:
        shape = (batch_size, 1, self.dim)

        times = torch.linspace(-1, self.num_timesteps - 1, steps=(self.sampling_timesteps or self.num_timesteps) + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=self.device)

        for time, time_next in tqdm(time_pairs, desc="DDIM Sampling Loop Time Step"):
            time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)

            # 1) Predict v and convert to eps
            v = self.model(x, time_cond)
            eps = self.predict_noise_from_v(x, time_cond, v)

            # 2) Guidance: eps_hat = eps − sqrt(1 − alpha_bar_t) * del_{x_t} log f(y|x_t)
            alpha_bar_t = extract(self.alphas_cumprod, time_cond, x.shape)
            if cond_fn is not None:
                _x0_hat = self.predict_start_from_noise(x, time_cond, eps)
                grad = F.normalize(cond_fn(_x0_hat, time_cond), dim=-1)
                eps_hat = eps - (1.0 - alpha_bar_t).sqrt() * grad * guidance_scale
            else:
                eps_hat = eps

            # 3) Compute x0_hat
            x0_hat = self.predict_start_from_noise(x, time_cond, eps_hat)

            # 4) DDIM update
            if time_next < 0:
                x = x0_hat
                continue

            time_next_cond = torch.full((batch_size,), time_next, device=self.device, dtype=torch.long)
            alpha_bar_next = extract(self.alphas_cumprod, time_next_cond, x.shape)

            x = alpha_bar_next.sqrt() * x0_hat + (1.0 - alpha_bar_next).sqrt() * eps_hat

        return x.reshape(batch_size, self.dim) if reshape_output else x
    
class DDPMSampler(DiffusionSampler):
    def __init__(
        self,
        diffusion_model : LatentDiffusionModel,
    ) -> None:
        super().__init__(diffusion_model=diffusion_model)

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
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

        #pred_v
        v = model_output
        x_start = self.predict_start_from_v(x, t, v)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

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
    def sample(
        self,
        batch_size: int = 16,
        guidance_scale: float = 1.0,
        cond_fn: Callable | None = None,
        reshape_output=True,
    ) -> torch.Tensor:
        shape = (batch_size, 1, self.dim)
        x = torch.randn(shape, device=self.device)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'DDPM Sampling Loop Time Step', total=self.num_timesteps):
            x, x_start = self.p_sample(x, t, batch_size=batch_size, cond_fn=cond_fn)

        return x.reshape(batch_size, self.dim) if reshape_output else x

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


        