# Installation
**N.B.:** You may need to make sure you have OpenMP installed on your machine before you can install Helix.

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

## Getting the code
You can obtain the Helix source code by cloning the repository from Github. Git may need to be downloaded onto your machine.
```shell
git clone https://github.com/Biomaterials-for-Medical-Devices-AI/Helix.git
```

## Setting up your developer environment
First you will need to ensure that you have Python installed. Helix requires version **3.11** or higher to run.

Next you need to create a virtual environment to run Helix. 

### Mac/Linux
```shell
# Create using venv
python -m venv <path/to/env>
source <path/to/env>/bin/activate

# -- OR --

# Create using conda
conda create -n <env_name> python=3.11
conda activate <env_name>
```

### Windows
```shell
# Create using venv
python -m venv <path\to\env>
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
N.B. py may work in place of python. 
Kill and Restart Terminal

```shell
<path/to/env>\Scripts\activate
```

# -- OR --

# Create using conda
conda create -n <env_name> python=3.11
conda activate <env_name>

### Install `uv`
Once you have activated your virtual environment, you need to install [uv](https://docs.astral.sh/uv/). To install `uv`, use the following command:

```shell
pip install uv
```

## Install Helix requirements
Deactivate the virtual environment using the command:
```shell
deactivate
```
To install the requirements for Helix, use the following commands:

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
This will prompt the user to add the local bin directory to their PATH. This can be achieved by:
1. Pressing the Win Button and search for Edit System Environment Variables
2. Click Environment Variables
3. Under User Variables, select PATH
4. Click Edit
5. Click New
6. Add local bin directory to PATH
Close all open terminals, and then reopen.
```shell
uv sync --all-groups
```

The `--all-groups` flag here will add the developer dependencies for formatting the code, code quality checks and testing.

## Running Helix
Once you have installed Helix, you can run it from the terminal like so:
```shell
uv run helix
```
A browser window will open to the main page of the app.