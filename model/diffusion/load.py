import torch
from .unet1d import Unet1D
from .diffusion1d import GaussianDiffusion1D

def create_diffusion_model(
        unet_dim=128,
        unet_dim_mults=(1, 2, 4, 8),

        diffusion_latent_dim=128,
        diffusion_timesteps=1000,
        sampling_timesteps=1000,
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

    # TODO: Can add sampling timesteps
    diffusion_model = GaussianDiffusion1D(
        model=unet_model,
        seq_length=diffusion_latent_dim,
        timesteps=diffusion_timesteps,
        objective='pred_v',
        sampling_timesteps=sampling_timesteps
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

    return diffusion_model

def load_diffusion_model(model, model_path, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    print(f"Loaded model successfully from {model_path}")

def sample_diffusion(model: GaussianDiffusion1D, batch_size=4):
    model.eval()
    with torch.no_grad():
        latents = model.sample(batch_size=batch_size)
        latents = latents.reshape(batch_size, -1)
        return latents

