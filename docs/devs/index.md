# Welcome to the BioFEFI developer documentation

## Code tour
The contents of the BioFEFI source code look like this:

```
├── .github
├── .gitignore
├── README.md
├── biofefi
├── docs
├── poetry.lock
├── pyproject.toml
└── static
```

### `.github`
This directory is files relating to Github Actions and templates for issues and pull requests.

### `.gitignore`
This file tells `git` to not track files and directories listed within.

### `README.md`
This file gives a quick overview of BioFEFI and is displayed on the main page of the repository.

### `biofefi`
This is where the application code lives.

### `docs`
This is the source directory of the documentation site.

### `poetry.lock`
This file tells `poetry` all the dependencies and their version numbers that need to be installed. When you run `poetry install`, `poetry` will check this file and download those versions of dependencies.

### `pyproject.toml`
This file specifies the metadata for the project, such as authors, version number and which packages to include. It is also where dependencies are specified. These include the app dependencies as well as the developer dependencies.

### `static`
This directory is for static assets, like images, to be used in the application.