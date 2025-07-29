from gdiffusion.diffusion.util import *
import torch

# abstract base class
class BetaSchedule:
    def get_betas(num_timesteps, *args):
        pass
    def get_schedule_name():
        pass

class BetaScheduleLinear(BetaSchedule):
    def get_betas(num_timesteps):
        """
        linear schedule, proposed in original ddpm paper
        """
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype = torch.float64)
    def get_schedule_name():
        return 'linear'

class BetaScheduleSigmoid(BetaSchedule):
    def get_betas(num_timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype = torch.float64) / num_timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    def get_schedule_name():
        return 'sigmoid'
    
class BetaScheduleCosine(BetaSchedule):
    def get_betas(num_timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype = torch.float64) / num_timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    def get_schedule_name():
        return 'cosine'