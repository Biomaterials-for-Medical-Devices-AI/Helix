# BioFEFI: Python Toolkit for Machine Learning, Feature Importance, and Fuzzy Interpretation

![License][license-badge]
![Python][python-badge]
![Poetry][poetry-badge]
![scikit-learn][sklearn-badge]
![Matplotlib][plt-badge]
![Linux][linux-badge]
![macOS][macos-badge]
![Windows][windows-badge]

![GitHub Issues or Pull Requests][issues-badge]
![Build docs status][build-docs-badge]
![Publish docs status][publish-docs-badge]
![Code quality status][code-quality-badge]
![PyPI downloads][downloads-badge]

## Overview

BioFEFI is a no-code application for training and interpreting machine learning models. You can search hyper-parameters manually or ask BioFEFI to perform hyper-parameter search automatically. Also included is the ability to perform exploratory data analysis before you create any models. BioFEFI also produces publication-ready graphs along with model metrics and, of course, the models themselves!

## Install and run BioFEFI

You will need to install **Python 3.11** or **3.12** to use BioFEFI. Make sure you also install `pip` (The Python package installer). If you don't already have it installed, [get Python.](https://www.python.org/downloads/)

You may need to make sure you have OpenMP installed on your machine before you can install BioFEFI. In the terminal use the following commands for your OS:

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

For information on how to install and run BioFEFI, check the [instructions](https://biomaterials-for-medical-devices-ai.github.io/BioFEFI/users/installation.html).

## Usage

BioFEFI will open in your internet browser when you run it. The main screen will appear giving a brief introduction to the app. To the left of the screen you will see a list of pages with the different functionalities of the app. Explanations can be found in the [instructions](https://biomaterials-for-medical-devices-ai.github.io/BioFEFI/index.html).

[poetry-badge]: https://img.shields.io/badge/Poetry-%233B82F6.svg?style=for-the-badge&logo=poetry&logoColor=0B3D8D
[sklearn-badge]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[plt-badge]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[linux-badge]: https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black
[macos-badge]: https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0
[windows-badge]: https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white
[python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[issues-badge]: https://img.shields.io/github/issues/Biomaterials-for-Medical-Devices-AI/BioFEFI?style=for-the-badge
[build-docs-badge]: https://img.shields.io/github/actions/workflow/status/Biomaterials-for-Medical-Devices-AI/BioFEFI/build-dcos.yml?style=for-the-badge&label=Build%20docs
[publish-docs-badge]: https://img.shields.io/github/actions/workflow/status/Biomaterials-for-Medical-Devices-AI/BioFEFI/publish-docs.yml?style=for-the-badge&label=Publish%20docs
[code-quality-badge]: https://img.shields.io/github/actions/workflow/status/Biomaterials-for-Medical-Devices-AI/BioFEFI/format-code.yml?style=for-the-badge&label=Code%20quality
[license-badge]: https://img.shields.io/github/license/Biomaterials-for-Medical-Devices-AI/BioFEFI?style=for-the-badge&label=License
[downloads-badge]: https://img.shields.io/pypi/dm/biofefi?style=for-the-badge

