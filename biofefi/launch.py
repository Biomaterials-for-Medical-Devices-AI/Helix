import subprocess
import os
import logging


def main():
    """The entrypoint of BioFEFI.
    This method shouldn't be called explicitly. Use the `biofefi` command in the
    terminal after installing the app.
    """
    app_path = os.path.join(os.path.dirname(__file__), "ui.py")
    try:
        subprocess.run(["streamlit", "run", app_path])
    except KeyboardInterrupt:
        logging.info("Shutting down BioFEFI...")
