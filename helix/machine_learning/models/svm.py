from sklearn.svm import SVC as SkLeanSVC
from sklearn.svm import SVR as SkLearnSVR


class SVC(SkLeanSVC):
    """A Helix implementation of scikit-learn's SVC.

    It is exactly the same, except it always sets `probability` to `True`
    to get access to the `predict_proba` method, and `max_iter` to 1000, similar
    to `LinearSVC`, to avoid hanging on infinte iterations.
    """

    def __init__(
        self,
        *,
        C=1,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0,
        shrinking=True,
        probability=True,  # set probability to True to access predict_proba
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=1000,  # set max_iter to 1000 to avoid infinite fitting loop
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )


class SVR(SkLearnSVR):
    """A Helix implementation of scikit-learn's SVR.

    It is exactly the same, except it always sets `max_iter` to 1000, similar
    to `LinearSVR`, to avoid hanging on infinte iterations.
    """

    def __init__(
        self,
        *,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0,
        tol=0.001,
        C=1,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=1000  # set max_iter to 1000 to avoid infinite fitting loop
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )
