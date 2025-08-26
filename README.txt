**Guided Diffusion Project Repo**

**Data / Dataset**
(hidden)
*data/*
- Data for the diffusion molecule and bayesopt

*datasets/* 
- Helper classes for various types of datasets

*saved_models/*
- Has the saved models already trained

**Guided Diffusion**
*gdiffusion/classifier/*
- This has the code for the logp and extinct classifiers, which are trained on the latent space values

*gdiffusion/diffusion/*
This has the code for the latent diffusion model
- diffusion.py has the code for the diffusion model wrapper and sampler
- diffusion_model.py has the code for the peptide / molecule specific parameters
- unet.py has the code for the unet
- beta_scheduler.py has the various beta schedulers, although we are only using Sigmoid for now
- util.py has the helper functions for the diffusion model

*gdiffusion/vae/*
This has the code for the VAE which converts the latent to Peptide / String (depending on the model).
- molformers has the code for the VAE itself
- vae.py has the code for a wrapper around the VAE for loading / encoding / decoding / sampling from the VAE

*gdiffusion/guidance.py*
- Helpers for guided diffusion


*scripts/*
- Various scripts that may or may not work. Useful as a reference though

*test/*
- Various test scripts to reference / test if implementation is working

*util/*
- Utility for plotting (visualization), stats, chem, etc.

