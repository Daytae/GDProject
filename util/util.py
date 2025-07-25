import torch

def get_device(device=None):
    if device is not None:
        return device
    
    return 'cuda' if torch.cuda.is_available() else 'cpu'

