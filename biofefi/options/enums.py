from enum import StrEnum


class ConfigStateKeys(StrEnum):
    """Enum containing the state key names for UI configuration inputs."""

    IsFeatureEngineering = "is_feature_engineering"
    # Machine Learning options
    IsMachineLearning = "is_machine_learning"
    ProblemType = "problem_type"
    DataSplit = "data_split"
    NumberOfBootstraps = "num_bootstraps"
    UseLinear = "use_linear"
    UseRandomForest = "use_rf"
    UseXGBoost = "use_xgb"
    Normalization = "normalization"
    ModelTypes = "model_types"
    SaveModels = "save_models"
    # rerun ML
    RerunML = "rerun_ml"
    # rerun FI
    RerunFI = "rerun_fi"
    # Feature Importance options
    IsFeatureImportance = "is_feature_importance"
    NumberOfImportantFeatures = "num_important_features"
    ScoringFunction = "scoring_function"
    NumberOfRepetitions = "num_repetitions"
    ShapDataPercentage = "shap_data_percentage"
    SaveFeatureImportanceOptions = "save_feature_importance_options"
    SaveFeatureImportanceResults = "save_feature_importance_results"
    LocalImportanceFeatures = "local_importance_methods"
    EnsembleMethods = "ensemble_methods"
    GlobalFeatureImportanceMethods = "global_feature_importance_methods"
    ExplainModels = "explain_models"
    ExplainAllModels = "explain_all_models"
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
    RandomSeed = "random_seed"
    MLLogBox = "ml_log_box"
    FILogBox = "fi_log_box"
    FuzzyLogBox = "fuzzy_log_box"
    UploadedModels = "uploaded_models"
    UseHyperParamSearch = "use_hyperparam_search"
    # Data Analysis options
    TargetVarDistribution = "target_var_distribution"
    ShowKDE = "show_kde"
    NBins = "n_bins"
    SaveTargetVarDistribution = "save_target_var_distribution"
    CorrelationHeatmap = "correlation_heatmap"
    DescriptorCorrelation = "descriptor_correlation"
    SelectAllDescriptorsCorrelation = "select_all_descriptors_correlation"
    SaveHeatmap = "save_heatmap"
    PairPlot = "pair_plot"
    SelectAllDescriptorsPairPlot = "select_all_descriptors_pair_plot"
    DescriptorPairPlot = "descriptor_pair_plot"
    SavePairPlot = "save_pair_plot"
    tSNEPlot = "tsne_plot"
    SaveTSNEPlot = "save_tsne_plot"


class ExecutionStateKeys(StrEnum):
    RunPipeline = "run_pipeline"


class OptimiserTypes(StrEnum):
    Adam = "adam"
    SGD = "sgd"
    RMSprop = "rmsprop"


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
    BRNNClassifier = "bayesianRegularised nn classifier"
    BRNNRegressor = "bayesianRegularised nn regressor"


class DataSplitMethods(StrEnum):
    Holdout = "holdout"
    KFold = "k-fold"
    NoSplit = "none"  # field name can't be None


class Metrics(StrEnum):
    Accuracy = "accuracy"
    F1Score = "f1_score"
    Precision = "precision_score"
    Recall = "recall_score"
    ROC_AUC = "roc_auc_score"
    R2 = "R2"
    MAE = "MAE"
    RMSE = "RMSE"


class PlotOptionKeys(StrEnum):
    AxisFontSize = "plot_axis_font_size"
    AxisTickSize = "plot_axis_tick_size"
    ColourScheme = "plot_colour_scheme"
    RotateXAxisLabels = "angle_rotate_xaxis_labels"
    RotateYAxisLabels = "angle_rotate_yaxis_labels"
    SavePlots = "save_plots"
    TitleFontSize = "plot_title_font_size"
    FontFamily = "plot_font_family"


class ViewExperimentKeys(StrEnum):
    ExperimentName = "view_experiment_name"
