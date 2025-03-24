"""This module contains choices that will appear in the UI."""

from helix.options.enums import (
    DataSplitMethods,
    Normalisations,
    ProblemTypes,
    SvmKernels,
    TransformationsY,
)

SVM_KERNELS = [
    SvmKernels.RBF.upper(),  # appear as RBF, not Rbf
    SvmKernels.Linear.capitalize(),
    SvmKernels.Poly.capitalize(),
    SvmKernels.Sigmoid.capitalize(),
]
PROBLEM_TYPES = [
    ProblemTypes.Classification.capitalize(),
    ProblemTypes.Regression.capitalize(),
]
NORMALISATIONS = [
    Normalisations.Standardisation.capitalize(),
    Normalisations.MinMax.capitalize(),
    Normalisations.NoNormalisation.capitalize(),
]
TRANSFORMATIONS_Y = [
    TransformationsY.Log.capitalize(),
    TransformationsY.Sqrt.capitalize(),
    TransformationsY.MinMaxNormalisation.capitalize(),
    TransformationsY.StandardisationNormalisation.capitalize(),
    TransformationsY.NoTransformation.capitalize(),
]

PLOT_FONT_FAMILIES = ["serif", "sans-serif", "cursive", "fantasy", "monospace"]
DATA_SPLITS = [
    DataSplitMethods.Holdout.capitalize(),
    DataSplitMethods.KFold.capitalize(),
]
