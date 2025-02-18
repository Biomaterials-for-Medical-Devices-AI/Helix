from dataclasses import dataclass

from biofefi.options.enums import Normalisations


@dataclass
class DataSplitOptions:
    """
    Options class specifying how to split data in bootstrapping or cross-validation,
    as well as the size of the test size.
    """

    n_bootstraps: int | None = None
    k_folds: int | None = None
    test_size: float = 0.2


@dataclass
class DataOptions:
    """
    Options class specifying where data are saved, how to split them, and which
    normalisation technique to use.
    """

    data_path: str | None = None
    data_split: DataSplitOptions | None = None
    normalisation: Normalisations = Normalisations.NoNormalisation
