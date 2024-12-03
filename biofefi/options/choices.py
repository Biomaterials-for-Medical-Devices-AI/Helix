from biofefi.options.enums import (
    DataSplitMethods,
    Normalisations,
    ProblemTypes,
    SvmKernels,
)

SVM_KERNELS = [
    SvmKernels.RBF.upper(),  # appear as RBF, not Rbf
    SvmKernels.Linear.capitalize(),
    SvmKernels.Poly.capitalize(),
    SvmKernels.Sigmoid.capitalize(),
    SvmKernels.Precomputed.capitalize(),
]
PROBLEM_TYPES = [
    ProblemTypes.Classification.capitalize(),
    ProblemTypes.Regression.capitalize(),
]
NORMALISATIONS = [
    Normalisations.Standardization.capitalize(),
    Normalisations.MinMax.capitalize(),
    Normalisations.NoNormalisation.capitalize(),
]
PLOT_FONT_FAMILIES = ["serif", "sans-serif", "cursive", "fantasy", "monospace"]
DATA_SPLITS = [
    DataSplitMethods.Holdout.capitalize(),
    DataSplitMethods.KFold.capitalize(),
]
