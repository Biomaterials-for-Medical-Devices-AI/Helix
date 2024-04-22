import argparse
import random
import pickle
import os

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets the seed for the experiment

    Parameters
    ----------
    seed: int
        The seed to use for the experiment
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def log_metrics(log_directory, opt: argparse.Namespace):
    ''' Log model or feature importance hyperparameters
   Parameters
    ----------
        log_directory: str
            The directory to save the log file
        opt: argparse.Namespace
            The options object
    Returns:
        None
    '''
    
    log_path =  os.path.join(log_directory, "metrics.txt")

    with open(log_path, 'w') as f:
        for arg in vars(opt):
            f.write(f'{arg}: {getattr(opt, arg)}\n')

    
            
