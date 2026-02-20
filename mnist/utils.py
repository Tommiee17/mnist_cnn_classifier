import random 
import torch
import numpy as np

def set_seed(seed = 1):
    # Define seed value
    seed = 1

    #Numpy seed
    np.random.seed(seed)
    
    # Seed python random module
    random.seed(seed)
    
    # Pytorch seed
    torch.manual_seed(seed)
    
    # Seed for Cuda, multi GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
