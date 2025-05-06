import logging
import os
import shutil
import subprocess
from pathlib import Path

from helix.utils.utils import create_directory


def main():
    """The entrypoint of Helix.
    This method shouldn't be called explicitly. Use the `helix` command in the
    terminal after installing the app.
    """
    app_path = os.path.join(os.path.dirname(__file__), "Home.py")
    source_config = Path(os.path.dirname(__file__)) / ".streamlit" / "config.toml"
    dest_config = Path.home() / ".streamlit"
    create_directory(dest_config)
    shutil.copy(source_config, dest_config)
    try:
        subprocess.run(["streamlit", "run", app_path])
    except KeyboardInterrupt:
        logging.info("Shutting down Helix...")
