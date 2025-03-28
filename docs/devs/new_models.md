# Adding a model to Helix

This page concerns what is required for a new model to be included into Helix. We will **not** accept models that do not meet this specification.

## Model specifications
Any model you add to Helix **must** conform to `scikit-learn`'s API.
This is to ensure that all models in Helix behave consistently without requiring checks on the capabilities of each individual model. This in turn keeps the complexity of the code base down.

Please see the `scikit-learn` docs to for details about their [API](https://scikit-learn.org/stable/developers/develop.html).

Currently we only support supervised machine learning algorithms in the form of classifiers or regressors. **Classifiers** must implement the [`ClassifierMixin`][ClassifierMixin] and **regressors** must implement the [`RegressorMixin`][RegressorMixin]. **Both** must implement the [`BaseEstimator`][BaseEstimator] class.

```python
class MyClassifier(ClassifierMixin, BaseEstimator):
    ...


class MyRegressor(RegressorMixin, BaseEstimator):
    ...
```

You **must** then override the `fit` and `predict` methods with the logic needed to fit your model and make predictions on data, respectively. 
For classifiers, you **must** also override the `predict_proba` method, which returns the probabilities for each class for each prediction. This is not a requirement of `scikit-learn` but of Helix.

```python
class MyClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        # perform fitting logic
        ...
        return self

    def predict(self, X):
        # perform prediction logic
        preds = ...
        return preds

    def predict_proba(self, X):
        # perform prediction logic and estimate class probabilities
        probs = ...
        return probs


class MyRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        # perform fitting logic
        ...
        return self

    def predict(self, X):
        # perform prediction logic
        preds = ...
        return preds
```

See [here](https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator) for more information.

Your model **must** be saveable as a pickle file (`.pkl`). This is how Helix persists the models it trains.

## Hyperparameter tuning
Your models **must** be designed with *manual* and *automatic* hyperparameter tuning in mind. Make sure your model's hyperparameters are set in the constructor of your class.

```python
class MyNewModel(ClassifierMixin, BaseEstimator):
    def __init__(self, param1: int, param2: float):
        self.param1 = param1
        self.param2 = param2
```
As per the notes in [`BaseEstimator`][BaseEstimator], do not include `*args`, and `**kwargs` to the constructor signature. The hyperparameters must be exhaustively set.

### Automatic hyperparameter search (AHPS)
For your model to work with AHPS, your model needs to have tunable hyperparameters. If there aren't any, it will **not** work with [`GridSearchCV`][GridSearchCV], which is how we perform AHPS.

You will need to create a search grid of your hyperparameters. AHPS uses this to find the best combination of hyperparameters for a model trained on a given dataset.

```python
MY_MODEL_GRID = {
    "param1": [1, 2, 3],
    "param2": [1.0, 1.1, 1.2]
}
```

Add your grid to `helix/options/search_grids.py`. 

You can make your grid as large or small as you like. It is not necessary to include all hyperparameters in the grid, either. The more hyperparameters you include, and the more values you add, the training process take longer to complete, so place consider the user experience when deciding how many parameters to train, and how many values to test.

### Manual hyperparameter search
You will need to create a form for users to input the values they wish to use for the hyperparameters of your model.

For each field, it would be helpful to include a help message explaining what the hyperparameter is and what it does.

### How to integrate your model into Helix
The following examples will show you how you can integrate your new model into Helix and make it available for users.

#### Create the form component
```python
# helix/components/forms.py
from helix.options.search_grids import MY_MODEL_GRID


# Create the form component for MyModel
def _my_model_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        param1 = st.number_input(
            "param1",
            value=1,
            help="""
            The first hyperparameter to my model.
            The bigger it is, ther more accuarate the model.
            """
        )
        param2 = st.number_input(
            "param2",
            value=0.1,
            max_value=1.0,
            min_value=0.0,
            help="""
            The second hyperparameter to my model.
            The closer the value to 1.0, the smarter it is.
            """
        )
        params = {
            "param1": param1,
            "param2": param2
        }
    else:
        params = MY_MODEL_GRID
    
    model_types["MY_MODEL"] = {
        "use": True,
        "params": params,
    }
    return model_types
```

#### Add the form component to the main form
To make your model selectable by the user, edit `ml_options_form` in `helix/components/forms.py` as shown below.

```python
# helix/components/forms.py

def ml_options_form():
    ...
    # Look for this to find where the models are made available
    st.subheader("Select and cofigure which models to train")
    ...
    # Add this underneath to make your model available
    if st.toggle("My Model", value=False):
        my_model_type = _my_model_opts(use_hyperparam_search)
        model_types.update(mymodel_type)
```

[BaseEstimator]: https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator
[ClassifierMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html
[RegressorMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html
[GridSearchCV]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
