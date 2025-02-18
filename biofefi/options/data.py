from dataclasses import dataclass

from biofefi.options.enums import Normalisations


@dataclass
class DataOptions:
    data_path: str | None = None
    data_split: dict | None = None
    dependent_variable: str | None = None
    normalisation: Normalisations = Normalisations.NoNormalisation
