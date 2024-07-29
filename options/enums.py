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
    # Feature Importance options
    IsFeatureImportance = "is_feature_importance"
    UsePermutation = "use_permutation"
    UseShap = "use_shap"
    UseMean = "use_mean"
    UseMajorityVote = "use_majority_vote"
    UseLime = "use_lime"
    UseLocalShap = "use_local_shap"
    NumberOfImportantFeatures = "num_important_features"
    ScoringFunction = "scoring_function"
    NumberOfRepetitions = "num_repetitions"
    ShapDataPercentage = "shap_data_percentage"
    # Fuzzy options
    FuzzyFeatureSelection = "fuzzy_feature_selection"
    NumberOfFuzzyFeatures = "num_fuzzy_features"
    GranularFeatures = "granular_features"
    NumberOfClusters = "num_clusters"
    ClusterNames = "cluster_names"
    NumberOfTopRules = "num_top_rules"
    # Base options
    ExperimentName = "experiment_name"
    DependentVariableName = "dependent_variable_name"
    UploadedFileName = "uploaded_file_name"
