from typing import Literal, NamedTuple
from collections import namedtuple

DiffusionObjective = Literal["pred_noise", "pred_v", "pred_x0"]

from datasets import LatentDataset
from model.diffusion.unet1d import Unet1D
from model.diffusion.latent_diffusion_model import LatentDiffusionModel
from model.diffusion.beta_scheduler import BetaScheduleCosine, BetaScheduleLinear, BetaScheduleSigmoid
from model.diffusion.load import create_diffusion_model, load_diffusion_model

