# Library imports
import gdiffusion as gd
import util

device = util.util.get_device()
print(f"device: {device}")

# peptide diffusion
DIFFUSION_PATH = "saved_models/peptide_model_v1-20.pt"

diffusion = gd.create_peptide_diffusion_model(DIFFUSION_PATH, device=device)
z = diffusion.sample(batch_size=2).reshape(-1, 256)
