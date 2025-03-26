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

#### Register your model name
In `helix/options/enums.py`, edit the `ModelNames` enum by adding your model name.

```python
class ModelNames(StrEnum):
    LinearModel = "linear model"
    RandomForest = "random forest"
    XGBoost = "xgboost"
    SVM = "svm"
    ...
    MyNewModel = "my new model
```

#### Making your model available to Helix
If your model is a classifier, edit `CLASSIFIERS` in `helix/options/choices/ml_models.py` by adding your model like so:

```python
# import your model

CLASSIFIERS: dict[ModelNames, type] = {
    ModelNames.LinearModel: LogisticRegression,
    ModelNames.RandomForest: RandomForestClassifier,
    ModelNames.XGBoost: XGBClassifier,
    ModelNames.SVM: SVC,
    ...
    ModelNames.MyNewModel: MyModel
}
```

If your model is a regressor, edit `REGRESSORS` in `helix/options/choices/ml_models.py` by adding your model like so:

```python
# import your model

REGRESSORS: dict[ModelNames, type] = {
    ModelNames.LinearModel: LinearRegression,
    ModelNames.RandomForest: RandomForestRegressor,
    ModelNames.XGBoost: XGBRegressor,
    ModelNames.SVM: SVR,
    ...
    ModelNames.MyNewModel: MyModel
}
```

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

## Documentation
Please add your new model to the user documentation. To do this, edit the the **"Options"** subsection of **"Selecting models to train"** in`docs/users/train_models.md`. This is a Markdown file, please see this [Markdown guide](https://www.markdownguide.org/getting-started/) for information on how to write using Markdown.

**If you do not document your model, your model will not be added to Helix.**

Your explanation **must** include the hyperparameters and explanations of what they do to your model. It should also include a brief explanation of the theory of the model and link to any relevant papers or documentation concerning the model.

### Example
> - **My New Model**
>
>> My model uses a super cool algorithm that optimises 2 parameters `param1` and `param2` to asymptotically approach Artificial General Intelligence (AGI).
>
>> The paper can be found at [link here].
>> - param1: The first hyperparameter to my model. The bigger it is, ther more accuarate the model.
>> - param2: The second hyperparameter to my model. The closer the value to 1.0, the smarter it is.

## Testing
You **must** test that your model works with Helix for it to be included. Helix uses [`pytest`](https://docs.pytest.org/en/stable/index.html) and `streamlit`'s [testing framework](https://docs.streamlit.io/develop/concepts/app-testing/get-started).

You **must** add a test for both automatic hyperparameter search and manual hyperparameter tuning.

### What to test
What you are testing, in this case, is not the performance of the model in terms of some metric like accuracy or R^2, but whether your model is properly integrated into Helix. Your tests should check the following:
- That there are no errors or exceptions when running the model
- That it creates the model directory in the experiment
- That it creates the expected `.pkl` file
- That it creates the plot directory for the experiment and that that directory is not empty. i.e. you get the performance plots
- That you get the file with the expected predictions
- That you get the file with the model metrics

### How to add tests
You should add your tests to `tests/pages/test_4_Train_Models.py`.

Generally, you will write 2 test functions: one to test your model with automatic hyperparameter search, and one to test it with manual hyperparameter tuning. Take the tests for SVM models. You will find 2 tests: `test_auto_svm` and `test_manual_svm`. You might call your tests: `test_auto_<model_name>` and `test_manual_<model_name>`.

#### Testing AHPS
This test simulates the user setting up the model to be trained with [`GridSearchCV`][GridSearchCV]. This test should take one parameter called `new_experiment` of type `str`.

Below is `test_auto_svm` as an expample:

```python
def test_auto_svm(new_experiment: str):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    k = 3
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Set the number of k-folds
    at.number_input[0].set_value(k).run()
    # Select SVM
    at.toggle[4].set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_model_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".pkl"), map(str, expected_model_dir.iterdir()))
    )  # directory is not empty
    assert expected_plot_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".png"), map(str, expected_plot_dir.iterdir()))
    )  # directory is not empty
    assert expected_preds_file.exists()
    assert expected_metrics_file.exists()
```

You should be able to create a copy of this example and rename it to `test_auto_<my_model>`, edit the index on line 17 of the code to the number of the toggle you used in the page. To find this, count from **0**, from the first toggle on the page, up to the toggle for your model. e.g. SVM is the 5th toggle so the test does `at.toggle[4].set_value(True).run()`. If you added your model's toggle directly under SVM's, you'd do `at.toggle[5].set_value(True).run()`.

#### Testing manual hyperparameter tuning
This test simulates the user setting up the model to be trained without AHPS. This test should take 3 parameters called `new_experiment` of type `str`, `data_split_method` of type `DataSplitMethods` and `holdout_or_k` of type `int`.

Below is `test_manual_svm` as an expample. The decorator above the function signature doesn't need to be altered; it causes the test to run the page with bootstrapping and cross-validation.

```python
@pytest.mark.parametrize(
    "data_split_method,holdout_or_k",
    [
        (DataSplitMethods.Holdout.capitalize(), 3),
        (DataSplitMethods.KFold.capitalize(), 3),
    ],
)
def test_manual_svm(
    new_experiment: str, data_split_method: DataSplitMethods, holdout_or_k: int
):
    # Arrange
    exp_dir = helix_experiments_base_dir() / new_experiment
    expected_model_dir = ml_model_dir(exp_dir)
    expected_plot_dir = ml_plot_dir(exp_dir)
    expected_preds_file = ml_predictions_path(exp_dir)
    expected_metrics_file = ml_metrics_path(exp_dir)
    at = AppTest.from_file("helix/pages/4_Train_Models.py", default_timeout=120)
    at.run()

    # Act
    # Select the experiment
    at.selectbox[0].select(new_experiment).run()
    # Unselect automatic hyperparameter search, which is on by default
    at.toggle[0].set_value(False).run()
    # Select the data split method
    at.selectbox[1].select(data_split_method).run()
    # Set the number of bootstraps / k-folds
    at.number_input[0].set_value(holdout_or_k).run()
    # Select SVM
    at.toggle[4].set_value(True).run()
    # Leave hyperparameters on their default values
    # Leave save models and plots as true to get the outputs
    # Click run
    at.button[0].click().run()

    # Assert
    assert not at.exception
    assert not at.error
    assert expected_model_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".pkl"), map(str, expected_model_dir.iterdir()))
    )  # directory is not empty
    assert expected_plot_dir.exists()
    assert list(
        filter(lambda x: x.endswith(".png"), map(str, expected_plot_dir.iterdir()))
    )  # directory is not empty
    assert expected_preds_file.exists()
    assert expected_metrics_file.exists()
```

Similar to the automatic hyperparameter search test, you should only need to adjust the line saying `at.toggle[4].set_value(True).run()` to point to the correct toggle. Again, to find this, count from **0**, from the first toggle on the page, up to the toggle for your model.

### Running the tests
The tests will run when you open a pull request to Helix. They will re-run everytime you push to that PR. You can also run them manually:

```bash
uv run pytests
```

Be patient, the tests can take several minutes. Your changes may affect other tests, so be aware.

[BaseEstimator]: https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator
[ClassifierMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html
[RegressorMixin]: https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html
[GridSearchCV]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV