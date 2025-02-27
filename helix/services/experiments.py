import os
from pathlib import Path

from helix.options.data import DataOptions
from helix.options.execution import ExecutionOptions
from helix.options.file_paths import (
    data_options_path,
    execution_options_path,
    fi_options_dir,
    fi_options_path,
    fi_plot_dir,
    fi_result_dir,
    fuzzy_options_path,
    fuzzy_plot_dir,
    fuzzy_result_dir,
    helix_experiments_base_dir,
    log_dir,
    plot_options_path,
)
from helix.options.plotting import PlottingOptions
from helix.services.configuration import save_options
from helix.utils.utils import create_directory, delete_directory, delete_file


def get_experiments(base_dir: Path | None = None) -> list[str]:
    """Get the list of experiments in the Helix experiment directory.

    If `base_dir` is not specified, the default from `biofefi_experiments_base_dir`
    is used

    Args:
        base_dir (Path | None, optional): Specify a base directory for experiments.
        Defaults to None.

    Returns:
        list[str]: The list of experiments.
    """
    # Get the base directory of all experiments
    if base_dir is None:
        base_dir = helix_experiments_base_dir()

    if not base_dir.exists():
        # if no experiments directory, return empty list
        return []
    experiments = os.listdir(base_dir)
    # Filter out hidden files and directories
    experiments = filter(lambda x: not x.startswith("."), experiments)
    # Filter out files
    experiments = filter(
        lambda x: os.path.isdir(os.path.join(base_dir, x)), experiments
    )
    return list(experiments)


def create_experiment(
    save_dir: Path,
    plotting_options: PlottingOptions,
    execution_options: ExecutionOptions,
    data_options: DataOptions,
):
    """Create an experiment on disk with it's global plotting options,
    execution options and data options saved as `json` files.

    Args:
        save_dir (Path): The path to where the experiment will be created.
        plotting_options (PlottingOptions): The plotting options to save.
        execution_options (ExecutionOptions): The execution options to save.
        data_options (DataOptions): The data options to save.
    """
    create_directory(save_dir)
    plot_file_path = plot_options_path(save_dir)
    save_options(plot_file_path, plotting_options)
    execution_file_path = execution_options_path(save_dir)
    save_options(execution_file_path, execution_options)
    data_file_path = data_options_path(save_dir)
    save_options(data_file_path, data_options)


def find_previous_fi_results(experiment_path: Path) -> bool:
    """Find previous feature importance results.

    Args:
        experiment_path (Path): The path to the experiment.

    Returns:
        bool: whether previous experiments exist or not.
    """

    directories = [
        fi_plot_dir(experiment_path),
        fi_result_dir(experiment_path),
        fi_options_dir(experiment_path),
        fuzzy_plot_dir(experiment_path),
        fuzzy_result_dir(experiment_path),
        fuzzy_options_path(experiment_path),
        fi_options_path(experiment_path),
        log_dir(experiment_path) / "fi",
        log_dir(experiment_path) / "fuzzy",
    ]

    return any([d.exists() for d in directories])


def delete_previous_fi_results(experiment_path: Path):
    """Delete previous feature importance results.

    Args:
        experiment_path (Path): The path to the experiment.
    """

    directories = [
        fi_plot_dir(experiment_path),
        fi_result_dir(experiment_path),
        fi_options_dir(experiment_path),
        fuzzy_plot_dir(experiment_path),
        fuzzy_result_dir(experiment_path),
        fuzzy_options_path(experiment_path),
        fi_options_path(experiment_path),
        log_dir(experiment_path) / "fi",
        log_dir(experiment_path) / "fuzzy",
    ]

    for directory in directories:
        if directory.exists():
            if directory.is_file():
                delete_file(directory)
            elif directory.is_dir():
                delete_directory(directory)
