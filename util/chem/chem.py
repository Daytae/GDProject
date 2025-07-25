import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_logp(smiles, invalid_token=None):
    ''' 
        Calculates logp on a list of smile strings or smile string
        Returns @invalid_token if the smiles string is somehow invalid
    '''
    if isinstance(smiles, str):
        smiles = list(smiles)
    
    logps = []
    for smile in smiles:
        new_logp = invalid_token

        try:
            mol = Chem.MolFromSmiles(smile)

            if mol is None:
                print(f"Error: Invalid SMILES string '{smile}'")
            else:
                new_logp = Descriptors.MolLogP(mol)

        except Exception as e:
            print(f"Error getting mol from smiles: {e}")
        
        logps.append(new_logp)
    return logps


def selfies_to_smiles(selfies):
    return [sf.decoder(selfie) for selfie in selfies]