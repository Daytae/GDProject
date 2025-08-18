import torch
from gdiffusion.diffusion.unet1d import Unet1D
from gdiffusion.diffusion.latent_diffusion_model import LatentDiffusionModel
from gdiffusion.diffusion.beta_scheduler import BetaScheduleSigmoid
from util.util import get_device
from gdiffusion.diffusion.ddim_sampler import DDIMSampler

def create_diffusion_model(
        unet_dim=128,
        unet_dim_mults=(1, 2, 4, 8),
        diffusion_latent_dim=128,
        diffusion_timesteps=1000,
        objective="pred_v",
        beta_schedule=BetaScheduleSigmoid,
        clip_denoised=False,
        clip_min=-1.0,
        clip_max=1.0,
        model_path=None,
        device=None
    ):

    device = get_device(device)
    '''
        Utility for creating a diffusion model with a built in UNet
    '''

    # create the unet for the diffusion model
    unet_model = Unet1D(
        dim=unet_dim, 
        dim_mults=unet_dim_mults, 
        channels=1
    ).to(device)

    diffusion_model = LatentDiffusionModel(
        model=unet_model,
        beta_schedule=beta_schedule,
        objective=objective,
        latent_dim=diffusion_latent_dim,
        num_timesteps=diffusion_timesteps,
        device=device,
        clip_denoised=clip_denoised,
        clip_min=clip_min,
        clip_max=clip_max
    ).to(device)

    # print basic info and what not to let user know model is created

    total_params = sum(p.numel() for p in diffusion_model.parameters())
    trainable_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    
    print("")
    print("Model created successfully")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Model size: {total_params * 4 / (1024**2):.1f} MB")
    
    try:
        device = next(diffusion_model.parameters()).device
        print(f"- Device: {device}")
    except StopIteration:
        print("No parameters found")
    
    print(f"- Model Name: {type(diffusion_model).__name__}")

    if model_path is not None:
        checkpoint = torch.load(model_path)
        diffusion_model.load_state_dict(checkpoint['model'])
        
    return diffusion_model

def load_diffusion_model(model, model_path, device=None):
    device = get_device(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model']).to(device)
    

    print(f"Loaded model successfully from {model_path}")


def create_peptide_diffusion_model(model_path, device=None):
    device = get_device(device)
    diffusion_model = create_diffusion_model(
        unet_dim=256, 
        diffusion_latent_dim=256, 
        objective='pred_v', 
        beta_schedule=BetaScheduleSigmoid, 
        clip_denoised=False, 
        clip_min=-3.0,  
        clip_max=3.0, 
        model_path=None, 
        device=device
    ).to(device)

    try:
        device = next(diffusion_model.parameters()).device
        print(f"- Device: {device}")
    except StopIteration:
        print("No parameters found")
    
    print(f"- Model Name: {type(diffusion_model).__name__}")

    if model_path is not None:
        checkpoint = torch.load(model_path)
        diffusion_model.load_state_dict(checkpoint['model'])
        
    return diffusion_model

def load_diffusion_model(model, model_path, device=None):
    device = get_device(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model']).to(device)
    

    print(f"Loaded model successfully from {model_path}")


def create_ddim_sampler(diffusion_model: LatentDiffusionModel, sampling_timesteps: int = 50, ddim_sampling_eta:float = 0.0, device='cuda'):
    ddim = DDIMSampler(
        model=diffusion_model.model.to(device),
        latent_dim=diffusion_model.latent_dim,
        num_timesteps=diffusion_model.num_timesteps,
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=ddim_sampling_eta
    ).to(device)
    return ddim
    
