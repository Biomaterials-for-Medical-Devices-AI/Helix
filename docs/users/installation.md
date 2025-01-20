# Installation and running
## Before you start...
BioFEFI is installed and run via the command line. You can find the terminal on your computer in the following ways:

**On Mac:** [How to find the terminal on Mac](https://support.apple.com/en-gb/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac)

**On Windows:** [How to find the terminal on Windows](https://learn.microsoft.com/en-us/windows/terminal/faq#how-do-i-run-a-shell-in-windows-terminal-in-administrator-mode)

**On Linux:** Since there are are lots of distributions of Linux, you will have to use a search engine (e.g. Google) or lookup the instructions for your particular distribution.

## Installation
### Pre-requirements
You will need to install **Python 3.11** or greater to use BioFEFI. Make sure you also install `pip` (The Python package installer). If you don't already have it installed, [get Python.](https://www.python.org/downloads/)

You may need to make sure you have OpenMP installed on your machine before you can install BioFEFI. In the terminal use the following commands for your OS:

On Mac:
```shell
brew install libomp
```

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
pip install biofefi

# -- OR --

# Create a virtual environment with conda
conda create -n <env_name> python=3.11
conda activate <env_name>
pip install biofefi
```

## Windows
```shell
# Create a virtual environment with venv
python -m venv <path\to\env>
<path/to/env>\Scripts\activate
pip install biofefi

# -- OR --

# Create a virtual environment with conda
conda create -n <env_name> python=3.11
conda activate <env_name>
pip install biofefi
```

## Running BioFEFI
Once you have installed BioFEFI, you can run it from the terminal like so:
```shell
biofefi
```
A browser window will open to the main page of the app.

<!-- insert image here -->