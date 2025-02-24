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
You can obtain the Helix source code by cloning the repository from Github.
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
<path/to/env>\Scripts\activate

# -- OR --

# Create using conda
conda create -n <env_name> python=3.11
conda activate <env_name>
```

### Install `poetry`
Once you have activated your virtual environment, you need to install [poetry](https://python-poetry.org/). To install poetry, use the following command:

```shell
pip install poetry
```

## Install Helix requirements
To install the requirements for Helix, use the following command:

```shell
poetry install
```

## Running Helix
Once you have installed Helix, you can run it from the terminal like so:
```shell
poetry run biofefi
```
A browser window will open to the main page of the app.