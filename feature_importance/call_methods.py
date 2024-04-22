import argparse
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def load_model(model_name, folder):
    '''Load a saved model
    Args:
        model_name: Name of the model
        folder: Folder where the model is saved
    Returns:
        model: Model object
    '''
    with open(f"{folder}{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        return model

def load_data(opt: argparse.Namespace):   
    raise NotImplementedError(f"Funtion load_data is not implemented")