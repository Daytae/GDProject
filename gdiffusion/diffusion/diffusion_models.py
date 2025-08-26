
from gdiffusion.diffusion.unet1d import Unet1D
from gdiffusion.diffusion.diffusion import LatentDiffusionModel

class MoleculeDiffusionModel(LatentDiffusionModel):
    def __init__(self, unet_state_dict_path : str = None, device='cuda'):
        latent_dim = 128

        unet = Unet1D(
            dim=latent_dim,
            dim_mults=(1, 2, 4, 8),
            channels=1
        ).to(device)


        super().__init__(
            model=unet,
            latent_dim=latent_dim,
            num_timesteps=1000,
            clip_denoised=False,
            clip_bounds=(-1.0, 1.0),
            device=device
        )

        self.load_unet(unet_state_dict_path)

        # Only apply state dict 

class PeptideDiffusionModel(LatentDiffusionModel):
    def __init__(self, unet_state_dict_path : str = None, device='cuda'):
        latent_dim = 256

        unet =  Unet1D(
            dim=latent_dim,
            dim_mults=(1, 2, 4, 8),
            channels=1
        ).to(device)

        super().__init__(
            model=unet,
            latent_dim=latent_dim,
            num_timesteps=1000,
            clip_denoised=False,
            clip_bounds=(-3.0, 3.0),
            device=device
        )

        self.load_unet(unet_state_dict_path)
    