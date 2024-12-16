import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from biofefi.options.choices import SVM_KERNELS
from biofefi.options.enums import (
    ConfigStateKeys,
    ExecutionStateKeys,
    PlotOptionKeys,
    ProblemTypes,
)
from biofefi.options.file_paths import biofefi_experiments_base_dir, ml_model_dir
from biofefi.services.ml_models import load_models


def data_upload_form():
    """
    The main form for BioFEFI where the user supplies the data
    and says where they want their experiment to be saved.
    """
    st.header("Data Upload")
    save_dir = _save_directory_selector()
    # If a user has tried to enter a destination to save an experiment, show it
    # if it's valid, else show some red text showing the destination and saying
    # it's invalid.
    if not _directory_is_valid(save_dir) and st.session_state.get(
        ConfigStateKeys.ExperimentName
    ):
        st.markdown(f":red[Cannot use {save_dir}; it already exists.]")
    else:
        st.session_state[ConfigStateKeys.ExperimentName] = save_dir.name
    st.text_input(
        "Name of the dependent variable", key=ConfigStateKeys.DependentVariableName
    )
    st.file_uploader(
        "Choose a CSV file", type="csv", key=ConfigStateKeys.UploadedFileName
    )
    if not st.session_state.get(ConfigStateKeys.IsMachineLearning, False):
        st.file_uploader(
            "Upload machine leaerning models",
            type="pkl",
            accept_multiple_files=True,
            key=ConfigStateKeys.UploadedModels,
        )
    st.button("Run", key=ExecutionStateKeys.RunPipeline)


def _save_directory_selector() -> Path:
    """Create a selector for the directory to save experiments."""
    root = biofefi_experiments_base_dir()

    col1, col2 = st.columns([0.3, 0.7], vertical_alignment="bottom")

    col1.text(f"{root}{os.path.sep}", help="Your experiment will be saved here")
    sub_dir = col2.text_input("Name of the experiment", placeholder="e.g. MyExperiment")

    return root / sub_dir


def _directory_is_valid(directory: Path) -> bool:
    """Determine if the directory supplied by the user is valid. If it already exists,
    it is invalid.

    Args:
        directory (Path): The path to check.

    Returns:
        bool: `True` if the directory doesn't already exist, else `False`
    """
    return not directory.exists()


@st.experimental_fragment
def fi_options_form():
    global_methods = {}

    st.write("### Global Feature Importance Methods")
    st.write(
        "Select global methods to assess feature importance across the entire dataset. "
        "These methods help in understanding overall feature impact."
    )

    use_permutation = st.checkbox(
        "Permutation Importance",
        help="Evaluate feature importance by permuting feature values.",
    )

    global_methods["Permutation Importance"] = {
        "type": "global",
        "value": use_permutation,
    }

    use_shap = st.checkbox(
        "SHAP",
        help="Apply SHAP (SHapley Additive exPlanations) for global interpretability.",
    )
    global_methods["SHAP"] = {"type": "global", "value": use_shap}

    st.session_state[ConfigStateKeys.GlobalFeatureImportanceMethods] = global_methods

    st.write("### Ensemble Feature Importance Methods")
    st.write(
        "Ensemble methods combine results from multiple feature importance techniques, "
        "enhancing robustness. Choose how to aggregate feature importance insights."
    )

    # global methods need to be set to perform ensemble methods
    ensemble_is_disabled = not (use_permutation or use_shap)
    if ensemble_is_disabled:
        st.warning(
            "You must configure at least one global feature importance method to perform ensemble methods.",
            icon="⚠",
        )
    ensemble_methods = {}
    use_mean = st.checkbox(
        "Mean",
        help="Calculate the mean importance score across methods.",
        disabled=ensemble_is_disabled,
    )
    ensemble_methods["Mean"] = use_mean
    use_majority = st.checkbox(
        "Majority Vote",
        help="Use majority voting to identify important features.",
        disabled=ensemble_is_disabled,
    )
    ensemble_methods["Majority Vote"] = use_majority

    st.session_state[ConfigStateKeys.EnsembleMethods] = ensemble_methods

    st.write("### Local Feature Importance Methods")
    st.write(
        "Select local methods to interpret individual predictions. "
        "These methods focus on explaining predictions at a finer granularity."
    )

    local_importance_methods = {}
    use_lime = st.checkbox(
        "LIME",
        help="Use LIME (Local Interpretable Model-Agnostic Explanations) for local interpretability.",
    )
    local_importance_methods["LIME"] = {"type": "local", "value": use_lime}
    use_local_shap = st.checkbox(
        "Local SHAP",
        help="Use SHAP for local feature importance at the instance level.",
    )
    local_importance_methods["SHAP"] = {
        "type": "local",
        "value": use_local_shap,
    }

    st.session_state[ConfigStateKeys.LocalImportanceFeatures] = local_importance_methods

    st.write("### Additional Configuration Options")

    # Number of important features
    st.number_input(
        "Number of most important features to plot",
        min_value=1,
        value=10,
        help="Select how many top features to visualise based on their importance score.",
        key=ConfigStateKeys.NumberOfImportantFeatures,
    )

    # Scoring function for permutation importance
    if (
        st.session_state.get(ConfigStateKeys.ProblemType, ProblemTypes.Auto).lower()
        == ProblemTypes.Regression
    ):
        scoring_options = [
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ]
    elif (
        st.session_state.get(ConfigStateKeys.ProblemType, ProblemTypes.Auto).lower()
        == ProblemTypes.Classification
    ):
        scoring_options = ["accuracy", "f1"]
    else:
        scoring_options = []

    st.selectbox(
        "Scoring function for permutation importance",
        scoring_options,
        help="Choose a scoring function to evaluate the model during permutation importance.",
        key=ConfigStateKeys.ScoringFunction,
    )

    # Number of repetitions for permutation importance
    st.number_input(
        "Number of repetitions for permutation importance",
        min_value=1,
        value=5,
        help="Specify the number of times to shuffle each feature for importance estimation.",
        key=ConfigStateKeys.NumberOfRepetitions,
    )

    # Percentage of data to consider for SHAP
    st.slider(
        "Percentage of data to consider for SHAP",
        0,
        100,
        100,
        help="Set the percentage of data used to calculate SHAP values.",
        key=ConfigStateKeys.ShapDataPercentage,
    )

    # Fuzzy Options
    st.write("### Fuzzy Feature Selection Options")
    st.write(
        "Activate fuzzy methods to capture interactions between features in a fuzzy rule-based system. "
        "Define the number of features, clusters, and granular options for enhanced interpretability."
    )

    # both ensemble_methods and local_importance_methods
    fuzzy_is_disabled = (not (use_lime or use_local_shap)) or (
        not (use_mean or use_majority)
    )
    if fuzzy_is_disabled:
        st.warning(
            "You must configure both ensemble and local importance methods to use fuzzy feature selection.",
            icon="⚠",
        )
    fuzzy_feature_selection = st.checkbox(
        "Enable Fuzzy Feature Selection",
        help="Toggle fuzzy feature selection to analyze feature interactions.",
        key=ConfigStateKeys.FuzzyFeatureSelection,
        disabled=fuzzy_is_disabled,
    )

    if fuzzy_feature_selection:

        st.number_input(
            "Number of features for fuzzy interpretation",
            min_value=1,
            value=5,
            help="Set the number of features for fuzzy analysis.",
            key=ConfigStateKeys.NumberOfFuzzyFeatures,
        )

        st.checkbox(
            "Granular features",
            help="Divide features into granular categories for in-depth analysis.",
            key=ConfigStateKeys.GranularFeatures,
        )

        st.number_input(
            "Number of clusters for target variable",
            min_value=2,
            value=3,
            help="Set the number of clusters to categorise the target variable for fuzzy interpretation.",
            key=ConfigStateKeys.NumberOfClusters,
        )

        st.text_input(
            "Names of clusters (comma-separated)",
            help="Specify names for each cluster (e.g., Low, Medium, High).",
            key=ConfigStateKeys.ClusterNames,
            value=", ".join(["very low", "low", "medium", "high", "very high"]),
        )

        st.number_input(
            "Number of top occurring rules for fuzzy synergy analysis",
            min_value=1,
            value=10,
            help="Set the number of most frequent fuzzy rules for synergy analysis.",
            key=ConfigStateKeys.NumberOfTopRules,
        )

    st.subheader("Select outputs to save")

    # Save options
    st.toggle(
        "Save feature importance options",
        help="Save the selected configuration of feature importance methods.",
        key=ConfigStateKeys.SaveFeatureImportanceOptions,
        value=True,
    )

    st.toggle(
        "Save feature importance results",
        help="Store the results from feature importance computations.",
        key=ConfigStateKeys.SaveFeatureImportanceResults,
        value=True,
    )


@st.experimental_fragment
def ml_options_form():
    """The form for setting up the machine learning pipeline."""
    st.subheader("Select and cofigure which models to train")

    try:
        trained_models = load_models(
            ml_model_dir(
                biofefi_experiments_base_dir()
                / st.session_state[ConfigStateKeys.ExperimentName]
            )
        )

        if trained_models:
            st.warning("You have trained models in this experiment.")
            st.checkbox(
                "Would you like to rerun the experiments? This will overwrite the existing models.",
                value=True,
                key=ConfigStateKeys.RerunML,
            )
        else:
            st.session_state[ConfigStateKeys.RerunML] = True

    except Exception:
        st.session_state[ConfigStateKeys.RerunML] = True

    if st.session_state[ConfigStateKeys.RerunML]:

        model_types = {}
        use_linear = st.toggle("Linear Model", value=False)
        if use_linear:

            st.write("Options:")
            fit_intercept = st.checkbox("Fit intercept")
            model_types["Linear Model"] = {
                "use": use_linear,
                "params": {
                    "fit_intercept": fit_intercept,
                },
            }
            st.divider()

        use_rf = st.toggle("Random Forest", value=False)
        if use_rf:

            st.write("Options:")
            n_estimators_rf = st.number_input(
                "Number of estimators", value=300, key="n_estimators_rf"
            )
            min_samples_split = st.number_input("Minimum samples split", value=2)
            min_samples_leaf = st.number_input("Minimum samples leaf", value=1)
            max_depth_rf = st.number_input("Maximum depth", value=6, key="max_depth_rf")
            model_types["Random Forest"] = {
                "use": use_rf,
                "params": {
                    "n_estimators": n_estimators_rf,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_depth": max_depth_rf,
                },
            }
            st.divider()

        use_xgb = st.toggle("XGBoost", value=False)
        if use_xgb:

            st.write("Options:")
            n_estimators_xgb = st.number_input(
                "Number of estimators", value=300, key="n_estimators_xgb"
            )
            max_depth_xbg = st.number_input(
                "Maximum depth", value=6, key="max_depth_xgb"
            )
            learning_rate = st.number_input("Learning rate", value=0.01)
            subsample = st.number_input("Subsample size", value=0.5)
            model_types["XGBoost"] = {
                "use": use_xgb,
                "params": {
                    "kwargs": {
                        "n_estimators": n_estimators_xgb,
                        "max_depth": max_depth_xbg,
                        "learning_rate": learning_rate,
                        "subsample": subsample,
                    }
                },
            }
            st.divider()

        use_svm = st.toggle("Support Vector Machine", value=False)
        if use_svm:

            st.write("Options:")
            kernel = st.selectbox("Kernel", options=SVM_KERNELS)
            degree = st.number_input("Degree", min_value=0, value=3)
            c = st.number_input("C", value=1.0, min_value=0.0)
            model_types["SVM"] = {
                "use": use_svm,
                "params": {
                    "kernel": kernel.lower(),
                    "degree": degree,
                    "C": c,
                },
            }
            st.divider()

        st.session_state[ConfigStateKeys.ModelTypes] = model_types
        st.subheader("Select outputs to save")
        st.toggle(
            "Save models",
            key=ConfigStateKeys.SaveModels,
            value=True,
            help="Save the models that are trained to disk?",
        )
        st.toggle(
            "Save plots",
            key=PlotOptionKeys.SavePlots,
            value=True,
            help="Save the plots to disk?",
        )


@st.experimental_fragment
def target_variable_dist_form(data, dep_var_name, data_analysis_plot_dir):
    """
    Form to create the target variable distribution plot.
    """

    show_kde = st.toggle("Show KDE", value=True, key=ConfigStateKeys.ShowKDE)
    n_bins = st.slider(
        "Number of Bins",
        min_value=5,
        max_value=50,
        value=10,
        key=ConfigStateKeys.NBins,
    )

    if st.checkbox(
        "Create Target Variable Distribution Plot",
        key=ConfigStateKeys.TargetVarDistribution,
    ):

        displot = sns.displot(data=data, x=data.columns[-1], kde=show_kde, bins=n_bins)
        displot.set(title=f"{dep_var_name} Distribution")

        st.pyplot(displot)

        if st.button("Save Plot", key=ConfigStateKeys.SaveTargetVarDistribution):

            displot.savefig(data_analysis_plot_dir / f"{dep_var_name}_distribution.png")
            plt.clf()
            st.success("Plot created and saved successfully.")


@st.experimental_fragment
def correlation_heatmap_form(data, data_analysis_plot_dir):
    """
    Form to create the correlation heatmap plot.
    """

    if st.toggle(
        "Select All Descriptors",
        value=False,
        key=ConfigStateKeys.SelectAllDescriptorsCorrelation,
    ):
        default_corr = list(data.columns[:-1])
    else:
        default_corr = []

    corr_descriptors = st.multiselect(
        "Select columns to include in the correlation heatmap",
        data.columns[:-1],
        default=default_corr,
        key=ConfigStateKeys.DescriptorCorrelation,
    )

    corr_data = data[corr_descriptors + [data.columns[-1]]]

    if len(corr_descriptors) < 1:
        st.warning(
            "Please select at least one descriptor to create the correlation heatmap."
        )

    if st.checkbox(
        "Create Correlation Heatmap Plot", key=ConfigStateKeys.CorrelationHeatmap
    ):

        corr = corr_data.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        _ = sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=0.3,
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            cbar_kws={"shrink": 0.5},
        )

        st.pyplot(fig)

        if st.button("Save Plot", key=ConfigStateKeys.SaveHeatmap):

            fig.savefig(data_analysis_plot_dir / "correlation_heatmap.png")
            plt.clf()
            st.success("Plot created and saved successfully.")


@st.experimental_fragment
def pairplot_form(data, data_analysis_plot_dir):
    """
    Form to create the pairplot plot.
    """

    if st.toggle(
        "Select All Descriptors",
        value=False,
        key=ConfigStateKeys.SelectAllDescriptorsPairPlot,
    ):
        default_corr = list(data.columns[:-1])
    else:
        default_corr = None

    descriptors = st.multiselect(
        "Select columns to include in the pairplot",
        data.columns[:-1],
        default=default_corr,
        key=ConfigStateKeys.DescriptorPairPlot,
    )

    pairplot_data = data[descriptors + [data.columns[-1]]]

    if len(descriptors) < 1:
        st.warning(
            "Please select at least one descriptor to create the correlation plot."
        )

    if st.checkbox("Create Pairplot", key=ConfigStateKeys.PairPlot):

        pairplot = sns.pairplot(pairplot_data, corner=True)
        st.pyplot(pairplot)

        if st.button("Save Plot", key=ConfigStateKeys.SavePairPlot):
            pairplot.savefig(data_analysis_plot_dir / "pairplot.png")
            plt.clf()
            st.success("Plot created and saved successfully.")


@st.experimental_fragment
def tSNE_plot_form(data, random_state, data_analysis_plot_dir):

    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    X = data.drop(columns=[data.columns[-1]])
    y = data[data.columns[-1]]

    X = StandardScaler().fit_transform(X)

    if st.checkbox("Create t-SNE Plot", key=ConfigStateKeys.tSNEPlot):

        tsne = TSNE(n_components=2, random_state=random_state)
        X_embedded = tsne.fit_transform(X)

        df = pd.DataFrame(X_embedded, columns=["x", "y"])
        df["target"] = y

        fig = plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df, x="x", y="y", hue="target", palette="viridis")
        plt.title("t-SNE Plot")
        plt.ylabel("t-SNE Component 2")
        plt.xlabel("t-SNE Component 1")
        st.pyplot(fig)

        if st.button("Create and Save Plot", key=ConfigStateKeys.SaveTSNEPlot):

            fig.savefig(data_analysis_plot_dir / "tsne_plot.png")
            plt.clf()
            st.success("Plots created and saved successfully.")
