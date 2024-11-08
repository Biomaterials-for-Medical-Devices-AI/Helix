import os
from biofefi.options.file_paths import biofefi_experiments_base_dir


def get_experiments() -> list[str]:
    """Get the list of experiments in the BioFEFI experiment directory.

    Returns:
        list[str]: The list of experiments.
    """
    # Get the base directory of all experiments
    base_dir = biofefi_experiments_base_dir()
    experiments = os.listdir(base_dir)
    # Filter out hidden files and directories
    experiments = filter(lambda x: not x.startswith("."), experiments)
    # Filter out files
    experiments = filter(
        lambda x: os.path.isdir(os.path.join(base_dir, x)), experiments
    )
    return experiments
