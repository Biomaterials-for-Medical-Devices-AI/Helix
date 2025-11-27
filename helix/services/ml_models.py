import json
from pathlib import Path
from pickle import dump, load
from typing import TypeVar

from kan import KAN
from pandas import DataFrame
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from helix.machine_learning.models.KANs import KANMixin
from helix.options.choices.ml_models import CLASSIFIERS, REGRESSORS
from helix.options.enums import ProblemTypes
from helix.utils.utils import create_directory, get_trained_ml_models

MlModel = TypeVar("MlModel", BaseEstimator, ClassifierMixin, RegressorMixin)


def save_model_predictions(predictions: DataFrame, path: Path):
    """Save the predictions of the models to the given file path.

    Args:
        predictions (DataFrame): The predictions to save.
        path (Path): The file path to save the predictions.
    """
    create_directory(path.parent)

    predictions.to_csv(path, index=False)


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

    if isinstance(model, KAN):
        model.saveckpt(path=path.with_suffix(""))
    else:
        with open(path, "wb") as f:
            dump(model, f, protocol=5)


def load_trained_models(path: Path, model_names: list | str) -> dict[str, list]:
    """Load pre-trained machine learning models.

    Args:
        path (Path): The path to the directory where the models are saved.
        model_names (list[str] | str): The names of the models to explain, or "all".

    Returns:
        dict[str, list]: Dictionary mapping model class names to lists of loaded models.
    """

    models: dict[str, list] = {}

    # Decide which files to process
    files = (
        list(path.iterdir())
        if model_names == "all"
        else [path / name for name in model_names]
    )

    def add_model_to_dict(model):
        """Helper to add a model to the dictionary."""
        name = model.__class__.__name__
        models.setdefault(name, []).append(model)

    for file_name in files:
        # try:
        if str(file_name).endswith(".pkl"):
            with open(file_name, "rb") as f:
                model = load(f)
            add_model_to_dict(model)
        else:
            model = KANMixin.loadckpt(file_name)
            add_model_to_dict(model)
    # except UnpicklingError:
    #     continue  # ignore bad pickle files
    return models


def load_models(path: Path, model_names="all") -> dict[str, list]:
    """Load pre-trained machine learning models.

    Args:
        path (Path): The path to the directory where the models are saved.

    Returns:
        dict[str, list]: The pre-trained models.
    """

    files = (
        list(path.iterdir())
        if model_names == "all"
        else [path / name for name in model_names]
    )

    models: dict[str, list] = dict()
    for file_name in files:

        model_name = str(file_name).split("/")[-1]
        if str(file_name).endswith(".pkl"):
            with open(file_name, "rb") as file:
                model = load(file)
                models[model_name] = model
        else:
            model = KANMixin.loadckpt(file_name)
            models[model_name] = model

    return models


def get_model_type(model_type: str, problem_type: ProblemTypes) -> type:
    """
    Fetch the appropriate type for a given model name based on the problem type.

    Args:
        model_type (dict): The kind of model.
        problem_type (ProblemTypes): Type of problem (classification or regression).

    Raises:
        ValueError: If a model type is not recognised or unsupported.

    Returns:
        type: The constructor for a machine learning model class.
    """
    if problem_type.lower() == ProblemTypes.Classification:
        model_class = CLASSIFIERS.get(model_type.lower())
    elif problem_type.lower() == ProblemTypes.Regression:
        model_class = REGRESSORS.get(model_type.lower())
    if not model_class:
        raise ValueError(f"Model type {model_type} not recognised")

    return model_class


def models_exist(path: Path) -> bool:
    try:
        trained_models = get_trained_ml_models(path)

        if trained_models:
            return True
        else:
            return False

    except Exception:
        return False


def get_model(model_type: type, model_params: dict = None) -> MlModel:
    """Get a new instance of the requested machine learning model.

    If the model is to be used in a grid search, specify `model_params=None`.

    Args:
        model_type (type): The Python type (constructor) of the model to instantiate.
        model_params (dict, optional): The parameters to pass to the model constructor. Defaults to None.

    Returns:
        MlModel: A new instance of the requested machine learning model.
    """

    return model_type(**model_params) if model_params is not None else model_type()
