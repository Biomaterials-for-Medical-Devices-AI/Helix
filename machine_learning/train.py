import os
import sys

from machine_learning.data import load_data
from machine_learning.ml_options import MLOptions
from utils.utils import set_seed


def run() -> None:
    '''
    Run the ML training pipeline
    '''

    opt = MLOptions().parse()
    seed = opt.random_state
    data_path = opt.data_path
    data_split = opt.data_split

    # Set seed for reproducibility
    set_seed(seed)

    # Load data
    data = load_data(data_path, data_split, seed)
    
    

    
    


    