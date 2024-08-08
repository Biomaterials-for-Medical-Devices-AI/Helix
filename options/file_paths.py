from pathlib import Path
import os

BASE_DIR = os.getenv("BASE_DIR", Path.home() / ".BioFEFI")
if not isinstance(BASE_DIR, Path):
    BASE_DIR = Path(BASE_DIR)


def uploaded_file_path(file_name: str, exeperiment_name: str) -> Path:
    """Create the full upload path for data file uploads.

    Args:
        file_name (str): The name of the file.
        exeperiment_name (str): The name of the experiment. This will be used
        to create a subdirectory with the this value.

    Returns:
        Path: The full upload path for the file.
    """
    return BASE_DIR / exeperiment_name / file_name


def log_dir(exeperiment_name: str) -> Path:
    """Create the full upload path for experiment log files.

    Args:
        exeperiment_name (str): The name of the experiment. This will be used
        to create a subdirectory with the this value.

    Returns:
        Path: The full path for the log directory.
    """
    return BASE_DIR / exeperiment_name / "logs"


def ml_plot_dir(exeperiment_name: str) -> Path:
    """Create the pull path to the directory to save Machine Learning plots.

    Args:
        exeperiment_name (str): The name of the experiment. This will be used
        to create a subdirectory with the this value.

    Returns:
        Path: The full path for the Machine Learning plot directory.
    """
    return BASE_DIR / exeperiment_name / "plots" / "ml"
