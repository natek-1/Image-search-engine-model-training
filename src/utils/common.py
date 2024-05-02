import torch
import numpy as np
import time

def set_seed(seed: int = 42):
    '''
    sets the seed across numpy and pytorch library so to keep training repeatable
    input seed: the random seed you want to set
    '''
    torch.mps.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_unique_filename(filename: str, ext: str):
    '''
    generates a unique file name based on a root file given
    input filename: base name for file
    input ext: the extension for the file
    output str: a file name that is unique (based on current time)
    '''
    return time.strftime(f"{filename}_%Y_%m_%d_%H_%M.{ext}")
