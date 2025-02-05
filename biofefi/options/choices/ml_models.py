from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from biofefi.machine_learning.nn_models import (
    BayesianRegularisedNNClassifier,
    BayesianRegularisedNNRegressor,
)
from biofefi.options.enums import ModelNames

CLASSIFIERS = {
    ModelNames.LinearModel: LogisticRegression,
    ModelNames.RandomForest: RandomForestClassifier,
    ModelNames.XGBoost: XGBClassifier,
    ModelNames.SVM: SVC,
    ModelNames.BRNNClassifier: BayesianRegularisedNNClassifier,
}

REGRESSORS = {
    ModelNames.LinearModel: LinearRegression,
    ModelNames.RandomForest: RandomForestRegressor,
    ModelNames.XGBoost: XGBRegressor,
    ModelNames.SVM: SVR,
    ModelNames.BRNNClassifier: BayesianRegularisedNNRegressor,
}
