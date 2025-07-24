# img display
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import io

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
