from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from helix.machine_learning.models.BRNN_classifier import BRNNClassifier
from helix.machine_learning.models.BRNN_regressor import BRNNRegressor
from helix.machine_learning.models.svm import SVC, SVR
from helix.options.enums import ModelNames

CLASSIFIERS: dict[ModelNames, type] = {
    ModelNames.LinearModel: LogisticRegression,
    ModelNames.RandomForest: RandomForestClassifier,
    ModelNames.XGBoost: XGBClassifier,
    ModelNames.SVM: SVC,
    ModelNames.BRNNClassifier: BRNNClassifier,
}

REGRESSORS: dict[ModelNames, type] = {
    ModelNames.LinearModel: LinearRegression,
    ModelNames.RandomForest: RandomForestRegressor,
    ModelNames.XGBoost: XGBRegressor,
    ModelNames.SVM: SVR,
    ModelNames.BRNNClassifier: BRNNRegressor,
}
