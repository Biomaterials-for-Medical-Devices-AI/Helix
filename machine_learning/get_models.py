from typing import Dict, List

from utils.utils import assert_model_param


def get_models(
    model_types: Dict[str, Dict], problem_type: str, logger: object = None
) -> List:
    models = {}
    # model_names = [model for model, model_dict in model_types.items() if model_dict["use"]]
    model_list = [
        (model_type, model["params"])
        for model_type, model in model_types.items()
        if model["use"]
    ]
    model_names = [model[0] for model in model_list]
    model_params = [model[1] for model in model_list]
    for model, model_param in model_list:
        if model.lower() == "linear model":
            from sklearn.linear_model import LinearRegression, LogisticRegression

            if problem_type.lower() == "classification":
                model_param = assert_model_param(
                    LogisticRegression, model_param, logger=logger
                )
                models[model] = LogisticRegression(**model_param)
            elif problem_type.lower() == "regression":
                model_param = assert_model_param(
                    LinearRegression, model_param, logger=logger
                )
                models[model] = LinearRegression(**model_param)
            else:
                raise ValueError(f"Model type {model} not recognized")

        elif model.lower() == "random forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            if problem_type.lower() == "classification":
                model_param = assert_model_param(
                    RandomForestClassifier, model_param, logger=logger
                )
                models[model] = RandomForestClassifier(**model_param)
            elif problem_type.lower() == "regression":
                model_param = assert_model_param(
                    RandomForestRegressor, model_param, logger=logger
                )
                models[model] = RandomForestRegressor(**model_param)
            else:
                raise ValueError(f"Model type {model} not recognized")

        else:
            raise ValueError(f"Model type {model} not recognized")
    return models
