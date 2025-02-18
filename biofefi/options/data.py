from dataclasses import dataclass

from biofefi.options.enums import Normalisations


@dataclass
class DataSplitOptions:
    n_bootstraps: int | None = None
    k_folds: int | None = None
    test_size: float = 0.2


@dataclass
class DataOptions:
    data_path: str | None = None
    data_split: DataSplitOptions | None = None
    dependent_variable: str | None = None
    normalisation: Normalisations = Normalisations.NoNormalisation
