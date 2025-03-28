# Using your models

Currently, there is no ability to use your models to make predictions on new data inside Helix. Therefore, you will need to create your own Python scripts to use them. You will need to be familiar with the `scikit-learn` library. Particularly, with how to make predictions with models.

Helix saves models from an experiment to `/home/<username>/HelixExperiments/<experiment name>/models` on **Linux**, `/Users/<username>/HelixExperiments/<experiment name>/models` on **MacOS**, and `C:\\Users\<username>\HelixExperiments\<experiment name>\models` on **Windows**. The models are saved as **pickle files** (`.pkl`). These models can be reused in your own code for making predictions.

The example on this page (see [Loading your models](#loading-your-models) and onwards) will assume an experiment that trained a classification model. But the principles are also applicable to regression models.

## Before you start...
Helix uses [`pickle`](https://docs.python.org/3/library/pickle.html) to save models for later use. This means that to use the models again without warnings or, potentially, errors, you must use an environment with the same packages and versions as Helix. See this page on [model persistance](https://scikit-learn.org/stable/model_persistence.html#model-persistence).

To get a similar environment to that used by Helix, you will need to install the same versions of some packages. The easiest way to do this is to install Helix in your environment and you will get all the same packages with the same versions as Helix. For more information on which packages and versions Helix uses, Look at Helix's [`pyproject.toml`](https://github.com/Biomaterials-for-Medical-Devices-AI/Helix/blob/main/pyproject.toml).

### Mac/Linux
```shell
# Create a virtual environment with venv
python -m venv <path/to/env>
source <path/to/env>/bin/activate
pip install helix

# -- OR --

# Create a virtual environment with conda
conda create -n <env_name> python=3.11  # or 3.12
conda activate <env_name>
pip install helix
```

You may need to try `python3` and `pip3` if `python` and `pip` do not work.

### Windows
```shell
# Create a virtual environment with venv
python -m venv <path\to\env>
<path/to/env>\Scripts\activate
pip install numpy==1.26.4 pandas==2.2.2 xgboost==2.1.0 scikit-learn==1.5.2 torch==2.5.1 # or torch==2.2.0 on intel MacOS

# -- OR --

# Create a virtual environment with conda
conda create -n <env_name> python=3.11 # or 3.12
conda activate <env_name>
pip install numpy==1.26.4 pandas==2.2.2 xgboost==2.1.0 scikit-learn==1.5.2 torch==2.5.1 # or torch==2.2.0 on intel MacOS
```

## Loading your models
```python
import pickle

path_to_model_file = "..."  # insert the path to your model

# Open the model file in "read bytes" mode
with open(path_to_model_file, "rb") as model_file:
    clf = pickle.load(model_file)
```

## Making predictions
When making predictions with you model, you need to take care that the new data for which you wish to make predictions are preprocessed the same way as your training data. 

If you preprocessed your data on the **Data Preprocessing** page, you can check how you preprocessed your data by selcting to your experiment on any of the following pages:
- Data Preprocessing
- Data Visualisation
- Train Models
- Feature Importance
- View Experiments
Under the experiment selection box, there will be a dropdown labelled "Show Experiment Options". Click this to expand and view your preprocessing options under the section labelled "Preprocessing Options".

When you run preprocessing, you get a file in your experiment directory ending in `_preprocessed.csv` or `_preprocessed.xlsx`. Look in this file to see which features made it through feature selection. If some features do not make it through, you will need to remove those features from your new data, or your model will not work.

```python
import pandas as pd

new_data = pd.read_csv("my_new_data.csv", header=0)

# Perform preprocessing on the new data using the same methodolody that you used to preprocess the training data in the original experiment.
preprocessed_data = ...

X = prepocessed_data.iloc[:, :-1]
y = prepocessed_data.iloc[:, -1]

# Make predictions
y_pred = clf.predict(X)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred)
```