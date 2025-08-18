from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import gdiffusion as gd
from gdiffusion.diffusion.beta_scheduler import BetaScheduleSigmoid

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionSampler(nn.Module):
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
        model: nn.Module,
        latent_dim: int,
        num_timesteps: int = 1000,
        device='cuda'
    ) -> None:
        super().__init__()

        self.model = model
        self.dim = latent_dim
        self.num_timesteps = num_timesteps
        self.device = device

        betas = BetaScheduleSigmoid.get_betas(num_timesteps=num_timesteps)  # type: ignore

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
        model: nn.Module,
        latent_dim: int,
        num_timesteps: int = 1000,
        sampling_timesteps: int | None = None,
        ddim_sampling_eta: float = 0.0,
    ) -> None:
        super().__init__(model, latent_dim, num_timesteps)
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        guidance_scale: float = 1.0,
        cond_fn: Callable | None = None,
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

        return x
    