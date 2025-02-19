from dataclasses import dataclass

from biofefi.options.enums import ProblemTypes


@dataclass
class ExecutionOptions:
    data_path: str | None = None
    experiment_name: str = "test"
    random_state: int = 1221
    problem_type: ProblemTypes = ProblemTypes.Classification
    dependent_variable: str | None = None
