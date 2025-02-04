from dataclasses import dataclass


@dataclass
class PreprocessingOptions:
    feature_selection_methods: dict
    variance_threshold: float
    correlation_threshold: float
    lasso_regularisation_term: float
    independent_variable_normalisation: str = "none"
    dependent_variable_transformation: str = "none"
