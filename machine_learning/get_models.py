from typing import Dict, List


def get_models(model_types: Dict[str, bool], problem_type: str) -> List:
    models = {}
    model_names = [model for model, use in model_types.items() if use]

    for model in model_names:
        if model.lower() == "linear model":
            from sklearn.linear_model import (LinearRegression,
                                              LogisticRegression)
            if problem_type.lower() == "classification":
                models[model] = LogisticRegression()
            elif problem_type.lower() == "regression":
                models[model] = LinearRegression()
            else:
                raise ValueError(f"Model type {model} not recognized")

        elif model.lower() == "random forest":
            from sklearn.ensemble import (RandomForestClassifier,
                                          RandomForestRegressor)
            if problem_type.lower() == "classification":
                models[model] = RandomForestClassifier()
            elif problem_type.lower() == "regression":
                models[model] = RandomForestRegressor()
            else:
                raise ValueError(f"Model type {model} not recognized")
            
        else:
            raise ValueError(f"Model type {model} not recognized")
        
    return models
            
        