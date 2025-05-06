# Installation and running
## Before you start...
Helix is installed and run via the command line. You can find the terminal on your computer in the following ways:

**On Mac:** [How to find the terminal on Mac](https://support.apple.com/en-gb/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac)

**On Windows:** [How to find the terminal on Windows](https://learn.microsoft.com/en-us/windows/terminal/faq#how-do-i-run-a-shell-in-windows-terminal-in-administrator-mode)

**On Linux:** Since there are are lots of distributions of Linux, you will have to use a search engine (e.g. Google) or lookup the instructions for your particular distribution.

## Installation
### Pre-requirements
You will need to install **Python 3.11** or **3.12** to use Helix. Make sure you also install `pip` (The Python package installer). If you don't already have it installed, [get Python.](https://www.python.org/downloads/)

You may need to make sure you have OpenMP installed on your machine before you can install Helix. In the terminal use the following commands for your OS:

On Mac:
```shell
brew install libomp
```

You may need to try `brew3` if `brew` does not work. Make sure you [install Homebrew](https://brew.sh/) on your Mac to use the `brew`/`brew3` command.

On Linux (Ubuntu)
```shell
sudo apt install libomp-dev
```

On Windows, this doesn't seem to be a problem. You should be able to proceed with installation.

---

## Mac/Linux
```shell
# Create a virtual environment with venv
python -m venv <path/to/env>
source <path/to/env>/bin/activate
pip install helix-ai

# -- OR --

# Create a virtual environment with conda
conda create -n <env_name> python=3.11  # or 3.12
conda activate <env_name>
pip install helix-ai
```

You may need to try `python3` and `pip3` if `python` and `pip` do not work.

## Windows
```shell
# Create a virtual environment with venv
python -m venv <path\to\env>
<path/to/env>\Scripts\activate
pip install helix-ai

# -- OR --

# Create a virtual environment with conda
conda create -n <env_name> python=3.11 # or 3.12
conda activate <env_name>
pip install helix-ai
```

## Running Helix
Once you have installed Helix, you can run it from the terminal like so:
```shell
helix
```
A browser window will open to the main page of the app.
