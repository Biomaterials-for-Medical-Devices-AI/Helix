from enum import StrEnum


class ConfigStateKeys(StrEnum):
    """Enum containing the state key names for UI configuration inputs."""

    IsFeatureEngineering = "is_feature_engineering"
    # Machine Learning options
    IsMachineLearning = "is_machine_learning"
    ProblemType = "problem_type"
    DataSplit = "data_split"
    NumberOfBootstraps = "num_bootstraps"
    SavePlots = "save_plots"
    UseLinear = "use_linear"
    UseRandomForest = "use_rf"
    UseXGBoost = "use_xgb"
    Normalization = "normalization"
    ModelTypes = "model_types"
    SaveModels = "save_models"
    # Feature Importance options
    IsFeatureImportance = "is_feature_importance"
    NumberOfImportantFeatures = "num_important_features"
    ScoringFunction = "scoring_function"
    NumberOfRepetitions = "num_repetitions"
    ShapDataPercentage = "shap_data_percentage"
    RotateXAxisLabels = "angle_rotate_xaxis_labels"
    RotateYAxisLabels = "angle_rotate_yaxis_labels"
    SaveFeatureImportancePlots = "save_feature_importance_plots"
    SaveFeatureImportanceOptions = "save_feature_importance_options"
    SaveFeatureImportanceResults = "save_feature_importance_results"
    LocalImportanceFeatures = "local_importance_methods"
    EnsembleMethods = "ensemble_methods"
    GlobalFeatureImportanceMethods = "global_feature_importance_methods"
    # Fuzzy options
    FuzzyFeatureSelection = "fuzzy_feature_selection"
    NumberOfFuzzyFeatures = "num_fuzzy_features"
    GranularFeatures = "granular_features"
    NumberOfClusters = "num_clusters"
    ClusterNames = "cluster_names"
    NumberOfTopRules = "num_top_rules"
    SaveFuzzySetPlots = "save_fuzzy_set_plots"
    # Base options
    ExperimentName = "experiment_name"
    DependentVariableName = "dependent_variable_name"
    UploadedFileName = "uploaded_file_name"
    RandomSeed = "random_seed"
    LogBox = "log_box"
    UploadedModels = "uploaded_models"


class ExecutionStateKeys(StrEnum):
    RunPipeline = "run_pipeline"


class ProblemTypes(StrEnum):
    Auto = "auto"
    Classification = "classification"
    Regression = "regression"


class SvmKernels(StrEnum):
    RBF = "rbf"
    Linear = "linear"
    Poly = "poly"
    Sigmoid = "sigmoid"
    Precomputed = "precomputed"


class Normalisations(StrEnum):
    Standardization = "standardization"
    MinMax = "minmax"
    NoNormalisation = "none"  # field name can't be None


class ModelNames(StrEnum):
    LinearModel = "linear model"
    RandomForest = "random forest"
    XGBoost = "xgboost"
    SVM = "svm"
