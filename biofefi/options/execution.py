from dataclasses import dataclass

from biofefi.options.enums import Normalisations, ProblemTypes


@dataclass
class ExecutionOptions:
    data_path: str | None = None
    data_split: dict | None = None
    experiment_name: str = "test"
    random_state: int = 1221
    problem_type: ProblemTypes = ProblemTypes.Classification
    dependent_variable: str | None = None
    normalization: Normalisations = Normalisations.NoNormalisation
    n_bootstraps: int = 3
