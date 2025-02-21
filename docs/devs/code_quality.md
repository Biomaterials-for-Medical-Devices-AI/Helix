# Code quality
## Style
Helix uses [`black`](https://black.readthedocs.io/en/stable/) to keep the code tidy, and consistent throughout. There is a Github Actions check on all pull requests which checks if the code format conforms to this style.

If you're code doesn't pass the format test, you can use the `black` command on the `biofefi` directory. `black` should be installed when you run `poetry install`.

```shell
poetry run black biofefi
```

## Quality
Helix uses `flake8` to check the quality of the code. It checks for unused variables and imports, as well as checking the complexity of functions. It also checks if imports are out of order. `flake8`'s output tells you where and what the problem is.

To check the code quality, run:
```shell
poetry run flake8 biofefi
```

To fix imports being out of order, use the `isort` command, which is installed when you run `poetry install` - along with `flake8`.

```shell
poetry run isort biofefi
```

To fix issues with complexity, you can look up the error online. See [here](https://www.flake8rules.com/) for the list of `flake8` rules.