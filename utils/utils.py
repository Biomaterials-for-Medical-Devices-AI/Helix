import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def set_seed(seed: int) -> None:
    """
    Sets the seed for the experiment

    Parameters
    ----------
    seed: int
        The seed to use for the experiment
    """
    # torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.cuda.manual_seed(seed)
