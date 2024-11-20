from pathlib import Path


def uploaded_file_path(file_name: str, experiment_path: Path) -> Path:
    """Create the full upload path for data file uploads.

    Args:
        file_name (str): The name of the file.
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full upload path for the file.
    """
    return experiment_path / file_name


def log_dir(experiment_path: Path) -> Path:
    """Create the full upload path for experiment log files.

    Args:
        experiment_path (str): The path of the experiment.

    Returns:
        Path: The full path for the log directory.
    """
    return experiment_path / "logs"


def ml_plot_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Machine Learning plots.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Machine Learning plot directory.
    """
    return experiment_path / "plots" / "ml"


def ml_model_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Machine Learning models.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Machine Learning model directory.
    """
    return experiment_path / "models"


def fi_plot_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Feature Importance plots.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Feature Importance plot directory.
    """
    return experiment_path / "plots" / "fi"


def fi_result_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Feature Importance results.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Feature Importance result directory.
    """
    return experiment_path / "results" / "fi"


def fi_options_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Feature Importance options.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Feature Importance options directory.
    """
    return experiment_path / "options" / "fi"


def fuzzy_plot_dir(experiment_path: Path) -> Path:
    """Create the full path to the directory to save Fuzzy plots.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Fuzzy plot directory.
    """
    return experiment_path / "plots" / "fuzzy"


def fuzzy_result_dir(experiment_path: str) -> Path:
    """Create the full path to the directory to save Fuzzy results.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The full path for the Fuzzy result directory.
    """
    return experiment_path / "results" / "fuzzy"


def biofefi_experiments_base_dir() -> Path:
    """Return the path the base directory of all BioFEFI experiments.

    This will be `/Users/<username>/BioFEFIExperiments` on MacOS,
    `/home/<username>/BioFEFIExperiments` on Linux, and
    `C:\\Users\\<username>\\BioFEFIExperiments` on Windows.

    Returns:
        Path: The path to the BioFEFI experiments base directory.
    """
    return Path.home() / "BioFEFIExperiments"


def plot_options_path(experiment_path: str) -> Path:
    """Return the path to an experiment's plot options.
    The path will be to a `json` file called `plot_options.json`

    Args:
        experiment_path (str): The path of the experiment.

    Returns:
        Path: The path to the experiment's plot options.

    Examples:
    ```python
    experiment_name = "test"
    experiment_path = biofefi_experiments_base_dir() / experiment_name
    plot_options_file = plot_options_path(experiment_path)
    ```
    """
    return experiment_path / "plot_options.json"


def execution_options_path(experiment_path: str) -> Path:
    """Return the path to an experiment's execution options.
    The path will be to a `json` file called `execution_options.json`

    Args:
        experiment_path (str): The path of the experiment.

    Returns:
        Path: The path to the experiment's execution options.

    Examples:
    ```python
    experiment_name = "test"
    experiment_path = biofefi_experiments_base_dir() / experiment_name
    exec_options_file = execution_options_path(experiment_path)
    ```
    """
    return experiment_path / "execution_options.json"
