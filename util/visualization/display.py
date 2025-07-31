# img display
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import io

import torch
import util.chem as chem
import util
import gdiffusion as gd
import util.stats as gdstats

def display_molecule(smiles_string, title=None):
    """
    Display molecular structure from SMILES string
    
    Args:
        smiles_string (str): SMILES notation of the molecule
        title (str): Optional title for the plot
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
        
        if title:
            plt.title(title, fontsize=14, fontweight='bold')
        else:
            plt.title(f'Molecule: {smiles_string}', fontsize=12)
        
        plt.tight_layout()
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
def display_latent(latent):
    latent = latent.cpu()
    latent_reshaped = torch.reshape(latent, shape = (8, -1))
    plt.figure(figsize=(8, 10))
    plt.imshow(latent_reshaped, cmap='viridis')

    # Add values to each cell
    for i in range(8):
        for j in range(latent_reshaped.shape[1]):
            plt.text(j, i, f'{latent_reshaped[i,j]:.2f}', 
                    ha='center', va='center', color='white')

    plt.title(label=f"Max Value: {latent.max()}  Min Value: {latent.min()}")
    plt.colorbar()
    plt.show()


def display_histograms(array_list, label_list):
    plt.figure(figsize=(10, 6))

    color_list = ['red', 'blue', 'green', 'yellow', 'pink', 'purple']
    for array, label, color in zip(array_list, label_list, color_list):
        plt.hist(array, alpha=0.7, label=label, density=True, color=color)

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Logps')
    plt.legend()
    plt.show()

# Utility functions:

def display_logp_info(z, z_diffusion_cached, vae, show_histogram=True, show_molecule=True, device=None):
    device = device if device is not None else vae.device

    logps_z = chem.latent_to_logp(z, vae=vae)
    logps_vae_normal = chem.latent_to_logp(torch.randn(size=(64, 128), device=z.device), vae=vae)
    logps_diffusion_cached = chem.latent_to_logp(z_diffusion_cached, vae=vae)

    if show_histogram:
        display_histograms([logps_z, logps_vae_normal, logps_diffusion_cached], ['Guided Diffusion LogPs', 'VAE Prior LogPs', 'Cached Diffusion LogPs'])
    
    # Example of one of the guided diffusion molecule:
    if show_molecule:
        smiles_gd = gd.latent_to_smiles(z, vae=vae)
        display_molecule(smiles_string=smiles_gd[0], title='Guided Diffusion Output')

    # you would not believe me but I actually wrote this not claude for once lmao

    print("LogP Stats: ")
    print(f"Max LogP: {max(logps_z):.2f}")
    print(f"Min LogP: {min(logps_z):.2f}")
    print(f"Avg LogP: {sum(logps_z) / len(logps_z):.2ff}")

    print("\nStats:")
    is_different, p_value = gdstats.is_different_from_other(z, z_diffusion_cached, do_print=True)
    