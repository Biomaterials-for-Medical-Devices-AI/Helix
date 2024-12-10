# Installation
**N.B.:** You may need to make sure you have OpenMP installed on your machine before you can install BioFEFI.

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
pip install git+https://github.com/Biomaterials-for-Medical-Devices-AI/BioFEFI.git

# -- OR --

# Create a virtual environment with conda
conda create -n <env_name> python=3.11
conda activate <env_name>
pip install git+https://github.com/Biomaterials-for-Medical-Devices-AI/BioFEFI.git
```

## Windows
```shell
# Create a virtual environment with venv
python -m venv <path\to\env>
<path/to/env>\Scripts\activate
pip install git+https://github.com/Biomaterials-for-Medical-Devices-AI/BioFEFI.git

# -- OR --

# Create a virtual environment with conda
conda create -n <env_name> python=3.11
conda activate <env_name>
pip install git+https://github.com/Biomaterials-for-Medical-Devices-AI/BioFEFI.git
```

## Running BioFEFI
Once you have installed BioFEFI, you can run it from the terminal like so:
```shell
biofefi
```
A browser window will open to the main page of the app.

<!-- insert image here -->