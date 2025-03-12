from dataclasses import dataclass

from helix.options.enums import ProblemTypes


@dataclass
class ExecutionOptions:
    experiment_name: str = "test"
    random_state: int = 1221
    problem_type: ProblemTypes = ProblemTypes.Classification
    dependent_variable: str | None = None
