import os
from pathlib import Path


def get_logs(log_dir: Path) -> str:
    """Get the latest log file for the latest run to display.

    Args:
        log_dir (Path): The directory to search for the latest logs.

    Raises:
        NotADirectoryError: `log_dir` does not point to a directory.

    Returns:
        str: The text of the latest log file.
    """
    if not log_dir.is_dir():
        raise NotADirectoryError(f"{log_dir} is not a directory")

    files = list(log_dir.iterdir())
    ctimes = [os.path.getctime(f) for f in files]
    most_recent = max(ctimes)
    index_most_recent = ctimes.index(most_recent)

    with open(files[index_most_recent], "r") as log:
        text = log.read()

    return text
