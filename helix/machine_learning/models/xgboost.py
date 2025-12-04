from xgboost import XGBClassifier as SkLearnXGBClassifier
from xgboost import XGBRegressor as SkLearnXGBRegressor


class XGBClassifier(SkLearnXGBClassifier):
    """A Helix implementation of xgboost's XGBClassifier.

    It is exactly the same, except it always sets `n_jobs` to 1
    so that when it runs in a parallel context (e.g. Grid Search),
    it doesn't slow down the training by consuming too much CPU
    """

    def __init__(self, *, objective="binary:logistic", n_jobs=1, **kwargs):
        super().__init__(objective=objective, n_jobs=n_jobs, **kwargs)


class XGBRegressor(SkLearnXGBRegressor):
    """A Helix implementation of xgboost's XGBRegressor.

    It is exactly the same, except it always sets `n_jobs` to 1
    so that when it runs in a parallel context (e.g. Grid Search),
    it doesn't slow down the training by consuming too much CPU
    """

    def __init__(self, *, objective="reg:squarederror", n_jobs=1, **kwargs):
        super().__init__(objective=objective, n_jobs=n_jobs, **kwargs)
