# img display
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import io
import os

import torch
import util.chem as chem
import util
import gdiffusion as gd
import util.stats as gdstats

def display_molecule(smiles_string, title=None, save_path=None):
    """
    Display molecular structure from SMILES string
    
    Args:
        smiles_string (str): SMILES notation of the molecule
        title (str): Optional title for the plot
        save_path (str or None): If set, save the image to this directory with a unique filename
    """
    try:
        # Parse SMILES string
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            print(f"Error: Invalid SMILES string '{smiles_string}'")
            return
        
        # Generate 2D coordinates for better visualization
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        
        # Create molecular image
        img = Draw.MolToImage(mol, size=(400, 400))
        
        # Convert PIL image to numpy array for matplotlib
        img_array = mpimg.pil_to_array(img)
        
        # Display the image
        plt.figure(figsize=(8, 6))
        plt.imshow(img_array)
        plt.axis('off')
        plt.title(title if title else f'Molecule: {smiles_string}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"molecule_vis.png"
            plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Print molecule information
        print(f"SMILES: {smiles_string}")
        # print(f"Molecular Formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
        # print(f"Molecular Weight: {Chem.rdMolDescriptors.CalcExactMolWt(mol):.2f}")
        # print(f"Number of Atoms: {mol.GetNumAtoms()}")
        # print(f"Number of Bonds: {mol.GetNumBonds()}")
        
    except ImportError as e:
        print("Error: Required packages not installed.")
        print("Install with: pip install rdkit matplotlib")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error processing molecule: {e}")

# Visualizes the latent space
def display_latent(latent, save_path=None):
    latent = latent.cpu()
    latent_reshaped = torch.reshape(latent, shape = (8, -1))
    plt.figure(figsize=(8, 10))
    plt.imshow(latent_reshaped, cmap='viridis')

    # Add values to each cell
    for i in range(8):
        for j in range(latent_reshaped.shape[1]):
            plt.text(j, i, f'{latent_reshaped[i,j]:.2f}', ha='center', va='center', color='white')

    plt.title(f"Max Value: {latent.max()}  Min Value: {latent.min()}")
    plt.colorbar()
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"latent_vis.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def display_histograms(array_list, label_list, save_path=None):
    plt.figure(figsize=(10, 6))
    color_list = ['red', 'blue', 'green', 'yellow', 'pink', 'purple']
    for array, label, color in zip(array_list, label_list, color_list):
        plt.hist(array, alpha=0.7, label=label, density=True, color=color)

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Logps')
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"histogram_vis.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Utility functions:
def latent_to_logp(z: torch.Tensor, vae : gd.MoleculeVAE):
    selfies = vae.decode(z)
    smiles = chem.selfies_to_smiles(selfies=selfies)
    logps = chem.calculate_logp(smiles, invalid_token=0.0)
    return logps

def display_logp_info(z, z_diffusion_cached, vae: gd.MoleculeVAE, show_histogram=True, show_molecule=True, device=None, save_path=None):
    device = device if device is not None else vae.device
    z_normal = torch.randn(size=(64, 128), device=z.device)

    logps_z = latent_to_logp(z, vae)
    logps_vae_normal = latent_to_logp(z_normal, vae)
    logps_diffusion_cached = latent_to_logp(z_diffusion_cached, vae)

    if show_histogram:
        display_histograms(
            [logps_z, logps_vae_normal, logps_diffusion_cached],
            ['Guided Diffusion LogPs', 'VAE Prior LogPs', 'Cached Diffusion LogPs'],
            save_path=save_path
        )

    if show_molecule:
        smiles_gd = chem.selfies_to_smiles(vae.decode(z))
        display_molecule(smiles_string=smiles_gd[0], title='Guided Diffusion Output', save_path=save_path)

    print("LogP Stats: ")
    print(f"Max LogP: {max(logps_z):.2f}")
    print(f"Min LogP: {min(logps_z):.2f}")
    print(f"Avg LogP: {sum(logps_z) / len(logps_z):.2f}")

    print("\nStats:")
    is_different, p_value = gdstats.is_different_from_other(z, z_diffusion_cached, do_print=True)
    