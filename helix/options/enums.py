from enum import StrEnum


class MachineLearningStateKeys(StrEnum):
    """Enum for the state keys related to machine learning."""

    ModelTypes = "model_types"
    SaveModels = "save_models"
    RerunML = "rerun_ml"
    MLLogBox = "ml_log_box"
    Predictions = "predictions"


class FeatureImportanceStateKeys(StrEnum):
    """Enum for the state keys related to feature importance."""

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
    FILogBox = "fi_log_box"


class FuzzyStateKeys(StrEnum):
    """Enum for the state keys related to fuzzy importance."""

    FuzzyFeatureSelection = "fuzzy_feature_selection"
    NumberOfFuzzyFeatures = "num_fuzzy_features"
    GranularFeatures = "granular_features"
    NumberOfClusters = "num_clusters"
    ClusterNames = "cluster_names"
    NumberOfTopRules = "num_top_rules"
    RerunFI = "rerun_fi"
    FuzzyLogBox = "fuzzy_log_box"


class ExecutionStateKeys(StrEnum):
    """Enum for state keys related to the execution of experiments."""

    ExperimentName = "experiment_name"
    DependentVariableName = "dependent_variable_name"
    UploadedFileName = "uploaded_file_name"
    RandomSeed = "random_seed"
    UseHyperParamSearch = "use_hyperparam_search"
    ProblemType = "problem_type"
    DataSplit = "data_split"
    NumberOfBootstraps = "num_bootstraps"
    Normalisation = "normalisation"


class DataAnalysisStateKeys(StrEnum):
    """Enum for app state keys relating to the Data Visualisation page."""

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
    TSNEPlot = "tsne_plot"
    SelectNormTsne = "select_norm_tsne"
    Perplexity = "perplexity"
    SaveTSNEPlot = "save_tsne_plot"


class DataPreprocessingStateKeys(StrEnum):
    """Enum for app state keys relating to the Data Preprocessing page."""

    DependentNormalisation = "dependent_normalisation"
    IndependentNormalisation = "independent_normalisation"
    ProceedTransformation = "proceed_transformation"
    VarianceThreshold = "variance_threshold"
    ThresholdVariance = "threshold_variance"
    CorrelationThreshold = "correlation_threshold"
    ThresholdCorrelation = "threshold_correlation"
    LassoFeatureSelection = "lasso_feature_selection"
    RegularisationTerm = "regularisation_term"


class OptimiserTypes(StrEnum):
    Adam = "adam"
    SGD = "sgd"
    RMSprop = "rmsprop"


class ActivationFunctions(StrEnum):
    ReLU = "relu"
    Sigmoid = "sigmoid"
    Tanh = "tanh"
    LeakyReLU = "leaky_relu"


class ProblemTypes(StrEnum):
    Auto = "auto"
    Classification = "classification"
    Regression = "regression"
    BinaryClassification = "binary_classification"
    MultiClassification = "multi_classification"


class SvmKernels(StrEnum):
    RBF = "rbf"
    Linear = "linear"
    Poly = "poly"
    Sigmoid = "sigmoid"
    Precomputed = "precomputed"


class Normalisations(StrEnum):
    # changing spelling to UK here would make this not backwards-compatible
    Standardisation = "standardisation"
    MinMax = "minmax"
    NoNormalisation = "none"  # field name can't be None


class TransformationsY(StrEnum):
    Log = "log"
    Sqrt = "square-root"
    MinMaxNormalisation = "minmax"
    # changing spelling to UK here would make this not backwards-compatible
    StandardisationNormalisation = "standardisation"  # to match normalisations
    NoTransformation = "none"


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
    ColourMap = "plot_colour_map"
    RotateXAxisLabels = "angle_rotate_xaxis_labels"
    RotateYAxisLabels = "angle_rotate_yaxis_labels"
    SavePlots = "save_plots"
    TitleFontSize = "plot_title_font_size"
    FontFamily = "plot_font_family"
    DPI = "dpi"
    Height = "plot_height"
    Width = "plot_width"


class ViewExperimentKeys(StrEnum):
    ExperimentName = "view_experiment_name"
