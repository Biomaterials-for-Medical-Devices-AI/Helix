import json, dataclasses
from pathlib import Path

from biofefi.options.execution import ExecutionOptions


def save_execution_options(path: Path, options: ExecutionOptions):
    """Save experiment execution options as a `json` file at the given path.

    Args:
        path (Path): The path to save the options.
        options (ExecutionOptions): The options to save.
    """
    options_json = dataclasses.asdict(options)
    with open(path, "w") as json_file:
        json.dump(options_json, json_file)


def load_execution_options(path: Path) -> ExecutionOptions:
    """Load experiment execution options from the given path.
    The path will be to a `json` file containing the options.

    Args:
        path (Path): The path the `json` file containing the options.

    Returns:
        ExecutionOptions: The plotting options.
    """
    with open(path, "r") as json_file:
        options_json = json.load(json_file)
    options = ExecutionOptions(**options_json)
    return options
