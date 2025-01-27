import json
import os
from pathlib import Path
from pickle import UnpicklingError, dump, load

from biofefi.options.choices import MODEL_PROBLEM_CHOICES
from biofefi.options.enums import ProblemTypes
from biofefi.utils.utils import create_directory


def save_models_metrics(metrics: dict, path: Path):
    """Save the statistical metrics of the models to the given file path.

    Args:
        metrics (dict): The metrics to save.
        path (Path): The file path to save the metrics.
    """

    create_directory(path.parent)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


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


def get_models(
    model_types: dict[str, dict],
    problem_type: str,
    logger: object = None,
    use_params: bool = True,
    use_grid_search: bool = False,
) -> dict:
    """
    Constructs and initializes machine learning models
    based on the given configuration.

    Args:
        model_types (dict): Dictionary containing model types
        and their parameters.
        problem_type (str): Type of problem (
            classification or regression).
        logger (object): Logger object to log messages.
        use_params (bool, optional): Add the parameters to models or leave them blank. Defaults to True.

    Raises:
        ValueError: If a model type is not recognized or unsupported

    Returns:
        dict: A dictionary of initialized models where th
        keys are model names and the values are instances
        of the corresponding models.
    """
    models = {}
    model_list = [
        (model_type, model["params"])
        for model_type, model in model_types.items()
        if model["use"]
    ]
    for model, model_params in model_list:
        if model_class := MODEL_PROBLEM_CHOICES.get(
            (model.lower(), problem_type.lower())
        ):
            if problem_type.lower() == ProblemTypes.Classification:
                model_params["class_weight"] = (
                    ["balanced"] if use_grid_search else "balanced"
                )
            models[model] = model_class(**model_params) if use_params else model_class()
            logger.info(
                f"Using model {model_class.__name__} with parameters {model_params}"
            )

        else:
            raise ValueError(f"Model type {model} not recognized")
    return models


def models_exist(path: Path) -> bool:
    try:
        trained_models = load_models(path)

        if trained_models:
            return True
        else:
            return False

    except Exception:
        return False
