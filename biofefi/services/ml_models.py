from pathlib import Path
from pickle import UnpicklingError, dump, load
from biofefi.options.ml import MachineLearningOptions
import json, dataclasses
import os


def save_ml_options(path: Path, options: MachineLearningOptions):
    """Save plot options to a `json` file at the specified path.

    Args:
        path (Path): The path to the `json` file.
        options (MachineLearningOptions): The options to save.
    """
    options_json = dataclasses.asdict(options)
    with open(path, "w") as json_file:
        json.dump(options_json, json_file, indent=4)


def save_model(model, path: Path):
    """Save a machine learning model to the given file path.

    Args:
        model (_type_): The model to save. Must be picklable.
        path (Path): The file path to save the model.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "wb") as f:
        dump(model, f, protocol=5)


def load_models(path: Path) -> dict[str, list]:
    """Load pre-trained machine learning models.

    Args:
        path (Path): The path to the directory where the models are saved.

    Returns:
        dict[str, list]: The pre-trained models.
    """
    models: dict[str, list] = dict()
    for file_name in path.iterdir():
        try:
            with open(file_name, "rb") as file:
                model = load(file)
                model_name = model.__class__.__name__
                if model_name in models:
                    models[model_name].append(model)
                else:
                    models[model_name] = [model]
        except UnpicklingError:
            pass  # ignore bad files

    return models


def load_models_to_explain(path: Path, model_names: list) -> dict[str, list]:
    """Load pre-trained machine learning models.

    Args:
        path (Path): The path to the directory where the models are saved.
        model_names (str): The name of the models to explain.

    Returns:
        dict[str, list]: The pre-trained models.
    """
    models: dict[str, list] = dict()
    for file_name in path.iterdir():
        if os.path.basename(file_name) in model_names or model_names == "all":
            try:
                with open(file_name, "rb") as file:
                    model = load(file)
                    model_name = model.__class__.__name__
                    if model_name in models:
                        models[model_name].append(model)
                    else:
                        models[model_name] = [model]
            except UnpicklingError:
                pass  # ignore bad files
    return models
