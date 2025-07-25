import torch
from model.diffusion.unet1d import Unet1D
from model.diffusion.latent_diffusion_model import LatentDiffusionModel
from model.diffusion.beta_scheduler import BetaScheduleSigmoid

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
        model_path=None
    ):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    )

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
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    print(f"Loaded model successfully from {model_path}")
