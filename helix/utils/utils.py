import argparse
import os
import random
import shutil
from multiprocessing import Process
from pathlib import Path

import numpy as np


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


def log_options(log_directory, opt: argparse.Namespace):
    """Log model or feature importance hyperparameters
    Parameters
     ----------
         log_directory: str
             The directory to save the log file
         opt: argparse.Namespace
             The options object
     Returns:
         None
    """

    log_path = os.path.join(log_directory, "options.txt")

    with open(log_path, "w") as f:
        for arg in vars(opt):
            f.write(f"{arg}: {getattr(opt, arg)}\n")


def create_directory(path: Path):
    """Create a directory at the specified path. If intermediate directories
    don't already exist, create them. If the path already exists, no action
    is taken.

    Args:
        path (Path): The path the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_upload(file_to_upload: str, content: str, mode: str = "w"):
    """Save a file given to the UI to disk.

    Args:
        file_to_upload (str): The name of the file to save.
        content (str): The contents to save to the file.
        mode (str): The mode to write the file. e.g. "w", "w+", "wb", etc.
    """
    base_dir = os.path.dirname(file_to_upload)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    with open(file_to_upload, mode) as f:
        f.write(content)


def cancel_pipeline(p: Process):
    """Cancel a running pipeline.

    Args:
        p (Process): the process running the pipeline to cancel.
    """
    if p.is_alive():
        p.terminate()


def delete_directory(path: Path):
    """Delete a directory and all of its contents.

    Returns:
        None

    Args:
        path: The path to the log directory
    """
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)  # Remove the directory


def delete_file(path: Path):
    """Delete a file.

    Returns:
        None

    Args:
        path: The path to the log directory
    """
    if path.is_file():
        path.unlink()  # Remove the file
