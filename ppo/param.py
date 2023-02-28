### Store the parameters
import os
import torch

class Param:
    dtype = None
    device = None
    # get absolute dir
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, r'learned_models')
    data_dir = os.path.join(root_dir, r'data')

    def __init__(self, dtype=None, device=None):
        if dtype is not None:
            Param.dtype = dtype
        Param.device = device
    def get():
        return (Param.dtype, Param.device)
    
def from_numpy(n_array, dtype=None):
    if dtype is None:
        return torch.from_numpy(n_array).to(Param.device).type(Param.dtype)
    else:
        return torch.from_numpy(n_array).to(Param.device).type(dtype)