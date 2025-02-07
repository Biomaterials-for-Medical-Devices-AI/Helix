from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from biofefi.machine_learning.models.svm import BioFefiSVC, BioFefiSVR
from biofefi.machine_learning.nn_models import (
    BayesianRegularisedNNClassifier,
    BayesianRegularisedNNRegressor,
)
from biofefi.options.enums import ModelNames

CLASSIFIERS = {
    ModelNames.LinearModel: LogisticRegression,
    ModelNames.RandomForest: RandomForestClassifier,
    ModelNames.XGBoost: XGBClassifier,
    ModelNames.SVM: BioFefiSVC,
    ModelNames.BRNNClassifier: BayesianRegularisedNNClassifier,
}

REGRESSORS = {
    ModelNames.LinearModel: LinearRegression,
    ModelNames.RandomForest: RandomForestRegressor,
    ModelNames.XGBoost: XGBRegressor,
    ModelNames.SVM: BioFefiSVR,
    ModelNames.BRNNClassifier: BayesianRegularisedNNRegressor,
}
