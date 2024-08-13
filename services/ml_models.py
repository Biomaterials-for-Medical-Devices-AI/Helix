from pathlib import Path
from pickle import UnpicklingError, dump, load


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
                    models[model_name] = [model_name]
        except UnpicklingError:
            pass  # ignore bad files

    return models
