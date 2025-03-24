import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from helix.components.configuration import data_split_options_box
from helix.components.plot_editor import edit_plot_modal
from helix.options.choices.ui import NORMALISATIONS, SVM_KERNELS, TRANSFORMATIONS_Y
from helix.options.choices.ml_models import ModelNames
from helix.options.enums import (
    DataAnalysisStateKeys,
    DataPreprocessingStateKeys,
    ExecutionStateKeys,
    FeatureImportanceStateKeys,
    FuzzyStateKeys,
    MachineLearningStateKeys,
    Normalisations,
    PlotOptionKeys,
    ProblemTypes,
    TransformationsY,
)
from helix.options.plotting import PlottingOptions
from helix.options.search_grids import (
    LINEAR_MODEL_GRID,
    MLREM_GRID,
    RANDOM_FOREST_GRID,
    SVM_GRID,
    XGB_GRID,
)


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

    st.session_state[FeatureImportanceStateKeys.GlobalFeatureImportanceMethods] = (
        global_methods
    )

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

    st.session_state[FeatureImportanceStateKeys.EnsembleMethods] = ensemble_methods

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

    st.session_state[FeatureImportanceStateKeys.LocalImportanceFeatures] = (
        local_importance_methods
    )

    st.write("### Additional Configuration Options")

    # Number of important features
    st.number_input(
        "Number of most important features to plot",
        min_value=1,
        value=10,
        help="Select how many top features to visualise based on their importance score.",
        key=FeatureImportanceStateKeys.NumberOfImportantFeatures,
    )

    # Scoring function for permutation importance
    if (
        st.session_state.get(ExecutionStateKeys.ProblemType, ProblemTypes.Auto).lower()
        == ProblemTypes.Regression
    ):
        scoring_options = [
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ]
    elif (
        st.session_state.get(ExecutionStateKeys.ProblemType, ProblemTypes.Auto).lower()
        == ProblemTypes.Classification
    ):
        scoring_options = ["accuracy", "f1"]
    else:
        scoring_options = []

    st.selectbox(
        "Scoring function for permutation importance",
        scoring_options,
        help="Choose a scoring function to evaluate the model during permutation importance.",
        key=FeatureImportanceStateKeys.ScoringFunction,
    )

    # Number of repetitions for permutation importance
    st.number_input(
        "Number of repetitions for permutation importance",
        min_value=1,
        value=5,
        help="Specify the number of times to shuffle each feature for importance estimation.",
        key=FeatureImportanceStateKeys.NumberOfRepetitions,
    )

    # Percentage of data to consider for SHAP
    st.slider(
        "Percentage of data to consider for SHAP",
        0,
        100,
        100,
        help="Set the percentage of data used to calculate SHAP values.",
        key=FeatureImportanceStateKeys.ShapDataPercentage,
    )

    # Fuzzy Options
    st.write("### Fuzzy Feature Importance Options")
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
    fuzzy_feature_importance = st.checkbox(
        "Enable Fuzzy Feature Importance",
        help="Toggle fuzzy feature importance to analyze feature interactions.",
        key=FuzzyStateKeys.FuzzyFeatureSelection,
        disabled=fuzzy_is_disabled,
    )

    if fuzzy_feature_importance:

        st.number_input(
            "Number of features for fuzzy interpretation",
            min_value=1,
            value=5,
            help="Set the number of features for fuzzy analysis.",
            key=FuzzyStateKeys.NumberOfFuzzyFeatures,
        )

        st.checkbox(
            "Granular features",
            help="Divide features into granular categories for in-depth analysis.",
            key=FuzzyStateKeys.GranularFeatures,
        )

        st.number_input(
            "Number of clusters for target variable",
            min_value=2,
            value=5,
            help="Set the number of clusters to categorise the target variable for fuzzy interpretation.",
            key=FuzzyStateKeys.NumberOfClusters,
        )

        st.text_input(
            "Names of clusters (comma-separated)",
            help="Specify names for each cluster (e.g., Low, Medium, High).",
            key=FuzzyStateKeys.ClusterNames,
            value=", ".join(["very low", "low", "medium", "high", "very high"]),
        )

        st.number_input(
            "Number of top occurring rules for fuzzy synergy analysis",
            min_value=1,
            value=10,
            help="Set the number of most frequent fuzzy rules for synergy analysis.",
            key=FuzzyStateKeys.NumberOfTopRules,
        )

    st.subheader("Select outputs to save")

    # Save options
    st.toggle(
        "Save feature importance options",
        help="Save the selected configuration of feature importance methods.",
        key=FeatureImportanceStateKeys.SaveFeatureImportanceOptions,
        value=True,
    )

    st.toggle(
        "Save feature importance results",
        help="Store the results from feature importance computations.",
        key=FeatureImportanceStateKeys.SaveFeatureImportanceResults,
        value=True,
    )


@st.experimental_fragment
def ml_options_form(problem_type: ProblemTypes = ProblemTypes.Regression):
    """
    The form for setting up the machine learning pipeline.
    """

    use_hyperparam_search = st.toggle(
        "Use hyper-parameter search",
        value=True,
        key=ExecutionStateKeys.UseHyperParamSearch,
    )
    if use_hyperparam_search:
        st.success(
            """
            **✨ Hyper-parameters will be searched automatically**.

            Helix will determine the best hyper-parameters
            and return the model with the best performance.
            """
        )
    else:
        st.info(
            """
            **🛠️ Manually set the hyper-parameters you wish to use for your models.**
            """
        )

    data_split_opts = data_split_options_box(not use_hyperparam_search)
    st.session_state[ExecutionStateKeys.DataSplit] = data_split_opts

    st.subheader("Select and cofigure which models to train")
    model_types = {}
    if st.toggle("Linear Model", value=False):
        lm_model_type = _linear_model_opts(use_hyperparam_search)
        model_types.update(lm_model_type)

    if st.toggle("Random Forest", value=False):
        rf_model_type = _random_forest_opts(use_hyperparam_search)
        model_types.update(rf_model_type)

    if st.toggle("XGBoost", value=False):
        xgb_model_type = _xgboost_opts(use_hyperparam_search)
        model_types.update(xgb_model_type)

    if st.toggle("Support Vector Machine", value=False):
        svm_model_type = _svm_opts(use_hyperparam_search)
        model_types.update(svm_model_type)

    # Only show MLREM for regression problems
    if problem_type == ProblemTypes.Regression:
        if st.toggle("Multiple Linear Regression with Expectation Maximisation", value=False):
            mlrem_model_type = _mlrem_opts(use_hyperparam_search)
            model_types.update(mlrem_model_type)

    st.session_state[MachineLearningStateKeys.ModelTypes] = model_types
    st.subheader("Select outputs to save")
    st.toggle(
        "Save models",
        key=MachineLearningStateKeys.SaveModels,
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
def target_variable_dist_form(
    data, dep_var_name, data_analysis_plot_dir, plot_opts: PlottingOptions
):
    """
    Form to create the target variable distribution plot.

    Uses plot-specific settings that are not saved between sessions.
    """

    show_kde = st.toggle("Show KDE", value=True, key=DataAnalysisStateKeys.ShowKDE)
    n_bins = st.slider(
        "Number of Bins",
        min_value=5,
        max_value=50,
        value=10,
        key=DataAnalysisStateKeys.NBins,
    )

    show_plot = st.checkbox(
        "Create Target Variable Distribution Plot",
        key=DataAnalysisStateKeys.TargetVarDistribution,
    )
    if show_plot or st.session_state.get("redraw_target_dist", False):
        if st.session_state.get("redraw_target_dist"):
            st.session_state["redraw_target_dist"] = False
            plt.close("all")  # Close any existing plots

        # Get plot-specific settings from session state or use loaded plot options
        plot_settings = st.session_state.get(
            "plot_settings_target_distribution",
            plot_opts,  # return the original plot_opts
        )

        plt.style.use(plot_settings.plot_colour_scheme)
        plt.figure(
            figsize=(
                plot_settings.width,
                plot_settings.height,
            ),
            dpi=plot_settings.dpi,
        )
        displot = sns.displot(
            data=data,
            x=data.columns[-1],
            kde=show_kde,
            bins=n_bins,
            height=plot_settings.height,
            aspect=plot_settings.width / plot_settings.height,
        )

        plt.title(
            f"{dep_var_name} Distribution",
            fontdict={
                "fontsize": plot_settings.plot_title_font_size,
                "family": plot_settings.plot_font_family,
            },
        )

        plt.xlabel(
            dep_var_name,
            fontsize=plot_settings.plot_axis_font_size,
            family=plot_settings.plot_font_family,
        )

        plt.ylabel(
            "Frequency",
            fontsize=plot_settings.plot_axis_font_size,
            family=plot_settings.plot_font_family,
        )

        plt.xticks(
            rotation=plot_settings.angle_rotate_xaxis_labels,
            fontsize=plot_settings.plot_axis_tick_size,
            family=plot_settings.plot_font_family,
        )
        plt.yticks(
            rotation=plot_settings.angle_rotate_yaxis_labels,
            fontsize=plot_settings.plot_axis_tick_size,
            family=plot_settings.plot_font_family,
        )

        st.pyplot(displot)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Save Plot", key=DataAnalysisStateKeys.SaveTargetVarDistribution
            ):
                displot.savefig(
                    data_analysis_plot_dir / f"{dep_var_name}_distribution.png"
                )
                plt.clf()
                st.success("Plot created and saved successfully.")
        with col2:
            if st.button(
                "Edit Plot",
                key=f"edit_{DataAnalysisStateKeys.SaveTargetVarDistribution}",
            ):
                st.session_state[
                    f"show_editor_{DataAnalysisStateKeys.SaveTargetVarDistribution}"
                ] = True

            if st.session_state.get(
                f"show_editor_{DataAnalysisStateKeys.SaveTargetVarDistribution}", False
            ):
                # Get plot-specific settings
                settings = edit_plot_modal(plot_opts, "target_distribution")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "Apply Changes", key="apply_changes_target_distribution"
                    ):
                        # Store settings in session state
                        st.session_state["plot_settings_target_distribution"] = settings
                        st.session_state[
                            "show_editor_{}".format(
                                DataAnalysisStateKeys.SaveTargetVarDistribution
                            )
                        ] = False
                        st.session_state["redraw_target_dist"] = True
                        st.rerun()
                with col2:
                    if st.button("Cancel", key="cancel_target_distribution"):
                        st.session_state[
                            "show_editor_{}".format(
                                DataAnalysisStateKeys.SaveTargetVarDistribution
                            )
                        ] = False
                        st.rerun()


@st.experimental_fragment
def correlation_heatmap_form(data, data_analysis_plot_dir, plot_opts: PlottingOptions):
    """
    Form to create the correlation heatmap plot.

    Uses plot-specific settings that are not saved between sessions.
    """

    if st.toggle(
        "Select All Descriptors",
        value=False,
        key=DataAnalysisStateKeys.SelectAllDescriptorsCorrelation,
    ):
        default_corr = list(data.columns[:-1])
    else:
        default_corr = []

    corr_descriptors = st.multiselect(
        "Select columns to include in the correlation heatmap",
        data.columns[:-1],
        default=default_corr,
        key=DataAnalysisStateKeys.DescriptorCorrelation,
    )

    corr_data = data[corr_descriptors + [data.columns[-1]]]

    if len(corr_descriptors) < 1:
        st.warning(
            "Please select at least one descriptor to create the correlation heatmap."
        )

    show_plot = st.checkbox(
        "Create Correlation Heatmap Plot", key=DataAnalysisStateKeys.CorrelationHeatmap
    )
    if show_plot or st.session_state.get("redraw_heatmap", False):
        if st.session_state.get("redraw_heatmap"):
            st.session_state["redraw_heatmap"] = False
            plt.close("all")  # Close any existing plots

        corr = corr_data.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Get plot-specific settings from session state or use loaded plot options
        plot_settings = st.session_state.get(
            "plot_settings_heatmap",
            plot_opts,
        )

        # Set up the matplotlib figure with the specified style
        plt.style.use(plot_settings.plot_colour_scheme)
        fig, ax = plt.subplots(
            figsize=(plot_settings.width, plot_settings.height),
            dpi=plot_settings.dpi,
        )

        # Draw the heatmap with enhanced styling
        sns.heatmap(
            corr,
            mask=mask,
            cmap=plot_settings.plot_colour_map,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=".2f",
            cbar_kws={
                "shrink": 0.5,
                "label": "Correlation Coefficient",
                "format": "%.1f",
                "aspect": 30,
                "drawedges": True,
            },
            annot_kws={
                "size": plot_settings.plot_axis_tick_size,
                "family": plot_settings.plot_font_family,
            },
            xticklabels=True,  # Ensure x-axis labels are shown
            yticklabels=True,  # Ensure y-axis labels are shown
            ax=ax,
        )

        # Customize the plot appearance
        ax.set_title(
            "Correlation Heatmap",
            fontsize=plot_settings.plot_title_font_size,
            family=plot_settings.plot_font_family,
            pad=20,  # Add padding above title
        )

        # Apply axis label rotations from plot settings
        plt.xticks(
            rotation=plot_settings.angle_rotate_xaxis_labels,
            ha="right",
            fontsize=plot_settings.plot_axis_tick_size,
            family=plot_settings.plot_font_family,
        )
        plt.yticks(
            rotation=plot_settings.angle_rotate_yaxis_labels,
            fontsize=plot_settings.plot_axis_tick_size,
            family=plot_settings.plot_font_family,
        )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Plot", key=DataAnalysisStateKeys.SaveHeatmap):
                fig.savefig(data_analysis_plot_dir / "correlation_heatmap.png")
                plt.clf()
                st.success("Plot created and saved successfully.")
        with col2:
            if st.button("Edit Plot", key=f"edit_{DataAnalysisStateKeys.SaveHeatmap}"):
                st.session_state[f"show_editor_{DataAnalysisStateKeys.SaveHeatmap}"] = (
                    True
                )

            if st.session_state.get(
                f"show_editor_{DataAnalysisStateKeys.SaveHeatmap}", False
            ):
                # Get plot-specific settings
                settings = edit_plot_modal(plot_opts, "heatmap")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Apply Changes", key="apply_changes_heatmap"):
                        # Store settings in session state
                        st.session_state["plot_settings_heatmap"] = settings
                        st.session_state[
                            f"show_editor_{DataAnalysisStateKeys.SaveHeatmap}"
                        ] = False
                        st.session_state["redraw_heatmap"] = True
                        st.rerun()
                with col2:
                    if st.button("Cancel", key="cancel_heatmap"):
                        st.session_state[
                            "show_editor_{DataAnalysisStateKeys.SaveHeatmap}"
                        ] = False
                        st.rerun()


@st.experimental_fragment
def pairplot_form(  # noqa: C901
    data, data_analysis_plot_dir, plot_opts: PlottingOptions
):
    """
    Form to create the pairplot plot.

    Uses plot-specific settings that are not saved between sessions.
    """

    if st.toggle(
        "Select All Descriptors",
        value=False,
        key=DataAnalysisStateKeys.SelectAllDescriptorsPairPlot,
    ):
        default_corr = list(data.columns[:-1])
    else:
        default_corr = None

    descriptors = st.multiselect(
        "Select columns to include in the pairplot",
        data.columns[:-1],
        default=default_corr,
        key=DataAnalysisStateKeys.DescriptorPairPlot,
    )

    pairplot_data = data[descriptors + [data.columns[-1]]]

    if len(descriptors) < 1:
        st.warning(
            "Please select at least one descriptor to create the correlation plot."
        )

    show_plot = st.checkbox("Create Pairplot", key=DataAnalysisStateKeys.PairPlot)
    if show_plot or st.session_state.get("redraw_pairplot", False):
        if st.session_state.get("redraw_pairplot"):
            st.session_state["redraw_pairplot"] = False
            plt.close("all")  # Close any existing plots

        # Get plot-specific settings from session state or use loaded plot options
        plot_settings = st.session_state.get(
            "plot_settings_pairplot",
            plot_opts,
        )

        # Set the style and create the pairplot
        plt.style.use(plot_settings.plot_colour_scheme)

        # Create figure with proper DPI
        with plt.rc_context({"figure.dpi": plot_settings.dpi}):
            # Calculate the figure size based on number of variables
            n_vars = len(pairplot_data.columns)
            aspect_ratio = plot_settings.width / plot_settings.height
            size_per_var = min(plot_settings.width, plot_settings.height) / n_vars

            pairplot = sns.pairplot(
                pairplot_data,
                height=size_per_var,
                aspect=aspect_ratio,
                corner=True,
                plot_kws={"s": 50, "alpha": 0.6},  # marker size  # transparency
                diag_kws={"bins": 20, "alpha": 0.6},
            )

            # Add title to the pairplot
            pairplot.figure.suptitle(
                "Pairplot",
                fontsize=plot_settings.plot_title_font_size,
                family=plot_settings.plot_font_family,
                y=1.02,  # Adjust title position to prevent overlap
            )

            # Apply axis label rotations and styling to all subplots
            for ax in pairplot.axes.flat:
                if ax is not None:  # Some axes might be None in corner=True mode
                    # Set x-axis label rotations
                    ax.set_xticklabels(
                        ax.get_xticklabels(),
                        rotation=plot_settings.angle_rotate_xaxis_labels,
                        family=plot_settings.plot_font_family,
                    )
                    # Set y-axis label rotations
                    ax.set_yticklabels(
                        ax.get_yticklabels(),
                        rotation=plot_settings.angle_rotate_yaxis_labels,
                        family=plot_settings.plot_font_family,
                    )

            # Customize the appearance after creating the plot
            for ax in pairplot.axes.flat:
                if ax is not None:
                    # Set font sizes
                    ax.tick_params(labelsize=plot_opts.plot_axis_tick_size, rotation=45)
                    if ax.get_xlabel():
                        ax.set_xlabel(
                            ax.get_xlabel(),
                            fontsize=plot_opts.plot_axis_font_size,
                            family=plot_opts.plot_font_family,
                        )
                    if ax.get_ylabel():
                        ax.set_ylabel(
                            ax.get_ylabel(),
                            fontsize=plot_opts.plot_axis_font_size,
                            family=plot_opts.plot_font_family,
                        )

            # Adjust layout to prevent label cutoff
            plt.tight_layout()
        st.pyplot(pairplot)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Plot", key=DataAnalysisStateKeys.SavePairPlot):
                pairplot.savefig(data_analysis_plot_dir / "pairplot.png")
                plt.clf()
                st.success("Plot created and saved successfully.")
        with col2:
            if st.button("Edit Plot", key=f"edit_{DataAnalysisStateKeys.SavePairPlot}"):
                st.session_state[
                    f"show_editor_{DataAnalysisStateKeys.SavePairPlot}"
                ] = True

            if st.session_state.get(
                f"show_editor_{DataAnalysisStateKeys.SavePairPlot}", False
            ):
                # Get plot-specific settings
                settings = edit_plot_modal(plot_opts, "pairplot")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Apply Changes", key="apply_changes_pairplot"):
                        # Store settings in session state
                        st.session_state["plot_settings_pairplot"] = settings
                        st.session_state[
                            "show_editor_{}".format(DataAnalysisStateKeys.SavePairPlot)
                        ] = False
                        st.session_state["redraw_pairplot"] = True
                        st.rerun()
                with col2:
                    if st.button("Cancel", key="cancel_pairplot"):
                        st.session_state[
                            "show_editor_{}".format(DataAnalysisStateKeys.SavePairPlot)
                        ] = False
                        st.rerun()


@st.experimental_fragment
def tSNE_plot_form(  # noqa: C901
    data,
    random_state,
    data_analysis_plot_dir,
    plot_opts: PlottingOptions,
    scaler: Normalisations = None,
):

    X = data.drop(columns=[data.columns[-1]])
    y = data[data.columns[-1]]

    if scaler == Normalisations.NoNormalisation:
        scaler = st.selectbox(
            "Select Normalisation for Comparison (this will not affect the normalisation for ML models)",
            options=[Normalisations.Standardisation, Normalisations.MinMax],
            key=DataAnalysisStateKeys.SelectNormTsne,
        )

    if scaler == Normalisations.MinMax:
        X_scaled = MinMaxScaler().fit_transform(X)
    elif scaler == Normalisations.Standardisation:
        X_scaled = StandardScaler().fit_transform(X)

    perplexity = st.slider(
        "Perplexity",
        min_value=5,
        max_value=50,
        value=30,
        help="The perplexity parameter controls the balance between local and global aspects of the data.",
        key=DataAnalysisStateKeys.Perplexity,
    )

    show_plot = st.checkbox("Create t-SNE Plot", key=DataAnalysisStateKeys.TSNEPlot)
    if show_plot or st.session_state.get("redraw_tsne", False):
        if st.session_state.get("redraw_tsne"):
            st.session_state["redraw_tsne"] = False
            plt.close("all")  # Close any existing plots

        tsne_normalised = TSNE(
            n_components=2, random_state=random_state, perplexity=perplexity
        )
        tsne_original = TSNE(
            n_components=2, random_state=random_state, perplexity=perplexity
        )

        X_embedded_normalised = tsne_normalised.fit_transform(X_scaled)
        X_embedded = tsne_original.fit_transform(X)

        df_normalised = pd.DataFrame(X_embedded_normalised, columns=["x", "y"])
        df_normalised["target"] = y

        df = pd.DataFrame(X_embedded, columns=["x", "y"])
        df["target"] = y

        # Get plot-specific settings from session state or use loaded plot options
        plot_settings = st.session_state.get(
            "plot_settings_tsne",
            plot_opts,
        )

        # Set style and create figure with proper DPI
        plt.style.use(plot_settings.plot_colour_scheme)
        with plt.rc_context({"figure.dpi": plot_settings.dpi}):
            fig, axes = plt.subplots(
                1, 2, figsize=(plot_settings.width, plot_settings.height)
            )

            # Plot 1: Normalised Data
            sns.scatterplot(
                data=df_normalised,
                x="x",
                y="y",
                hue="target",
                palette=plot_settings.plot_colour_map,
                s=100,  # marker size
                alpha=0.6,  # transparency
                ax=axes[0],
            )

            # Customize first plot
            axes[0].set_title(
                "t-SNE Plot (Normalised Features)",
                fontsize=plot_settings.plot_title_font_size,
                family=plot_settings.plot_font_family,
                pad=20,  # Add padding above title
            )
            axes[0].set_xlabel(
                "t-SNE Component 1",
                fontsize=plot_settings.plot_axis_font_size,
                family=plot_settings.plot_font_family,
            )
            axes[0].set_ylabel(
                "t-SNE Component 2",
                fontsize=plot_settings.plot_axis_font_size,
                family=plot_settings.plot_font_family,
            )
            # Apply axis label rotations and styling for first plot
            axes[0].tick_params(
                axis="both", which="major", labelsize=plot_settings.plot_axis_tick_size
            )
            for label in axes[0].get_xticklabels():
                label.set_rotation(plot_settings.angle_rotate_xaxis_labels)
                label.set_family(plot_settings.plot_font_family)
            for label in axes[0].get_yticklabels():
                label.set_rotation(plot_settings.angle_rotate_yaxis_labels)
                label.set_family(plot_settings.plot_font_family)

            # Plot 2: Original Data
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="target",
                palette=plot_settings.plot_colour_map,
                s=100,  # marker size
                alpha=0.6,  # transparency
                ax=axes[1],
            )

            # Customize second plot
            axes[1].set_title(
                "t-SNE Plot (Original Features)",
                fontsize=plot_settings.plot_title_font_size,
                family=plot_settings.plot_font_family,
                pad=20,  # Add padding above title
            )
            axes[1].set_xlabel(
                "t-SNE Component 1",
                fontsize=plot_settings.plot_axis_font_size,
                family=plot_settings.plot_font_family,
            )
            axes[1].set_ylabel(
                "t-SNE Component 2",
                fontsize=plot_settings.plot_axis_font_size,
                family=plot_settings.plot_font_family,
            )
            # Apply axis label rotations and styling for second plot
            axes[1].tick_params(
                axis="both", which="major", labelsize=plot_settings.plot_axis_tick_size
            )
            for label in axes[1].get_xticklabels():
                label.set_rotation(plot_settings.angle_rotate_xaxis_labels)
                label.set_family(plot_settings.plot_font_family)
            for label in axes[1].get_yticklabels():
                label.set_rotation(plot_settings.angle_rotate_yaxis_labels)
                label.set_family(plot_settings.plot_font_family)

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Plot", key=DataAnalysisStateKeys.SaveTSNEPlot):
                fig.savefig(data_analysis_plot_dir / "tsne_plot.png")
                plt.clf()
                st.success("Plots created and saved successfully.")
        with col2:
            if st.button("Edit Plot", key=f"edit_{DataAnalysisStateKeys.SaveTSNEPlot}"):
                st.session_state[
                    f"show_editor_{DataAnalysisStateKeys.SaveTSNEPlot}"
                ] = True

            if st.session_state.get(
                f"show_editor_{DataAnalysisStateKeys.SaveTSNEPlot}", False
            ):
                # Get plot-specific settings
                settings = edit_plot_modal(plot_opts, "tsne")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Apply Changes", key="apply_changes_tsne"):
                        # Store settings in session state
                        st.session_state["plot_settings_tsne"] = settings
                        st.session_state[
                            "show_editor_{}".format(DataAnalysisStateKeys.SaveTSNEPlot)
                        ] = False
                        st.session_state["redraw_tsne"] = True
                        st.rerun()
                with col2:
                    if st.button("Cancel", key="cancel_tsne"):
                        st.session_state[
                            "show_editor_{}".format(DataAnalysisStateKeys.SaveTSNEPlot)
                        ] = False
                        st.rerun()


def _linear_model_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        st.write("Options:")
        fit_intercept = st.checkbox("Fit intercept")
        params = {
            "fit_intercept": fit_intercept,
        }
        st.divider()
    else:
        params = LINEAR_MODEL_GRID
    model_types[ModelNames.LinearModel.value] = {
        "use": True,
        "params": params,
    }
    return model_types


def _random_forest_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        st.write("Options:")
        n_estimators_rf = st.number_input(
            "Number of estimators", value=100, key="n_estimators_rf"
        )
        min_samples_split = st.number_input("Minimum samples split", value=2)
        min_samples_leaf = st.number_input("Minimum samples leaf", value=1)
        col1, col2 = st.columns([0.25, 0.75], vertical_alignment="bottom", gap="small")
        use_rf_max_depth = col1.checkbox(
            "Set max depth",
            value=False,
            help="If disabled or 0, then nodes are expanded until all leaves are pure"
            " or until all leaves contain less than 'Minimum samples split'.",
        )
        max_depth_rf = col2.number_input(
            "Maximum depth",
            value="min",
            min_value=0,
            key="max_depth_rf",
            disabled=not use_rf_max_depth,
        )
        params = {
            "n_estimators": n_estimators_rf,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth_rf if max_depth_rf > 0 else None,
        }
        st.divider()
    else:
        params = RANDOM_FOREST_GRID
    model_types[ModelNames.RandomForest.value] = {
        "use": True,
        "params": params,
    }
    return model_types


def _xgboost_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        if st.checkbox("Set XGBoost options"):
            st.write("Options:")
            n_estimators_xgb = st.number_input(
                "Number of estimators", value=100, key="n_estimators_xgb"
            )
            learning_rate = st.number_input("Learning rate", value=0.01)
            subsample = st.number_input("Subsample size", value=0.5)
            col1, col2 = st.columns(
                [0.25, 0.75], vertical_alignment="bottom", gap="small"
            )
            use_xgb_max_depth = col1.checkbox(
                "Set max depth",
                value=False,
                help="If disabled or 0, then nodes are expanded until all leaves are pure.",
            )
            max_depth_xbg = col2.number_input(
                "Maximum depth",
                value="min",
                min_value=0,
                key="max_depth_xgb",
                disabled=not use_xgb_max_depth,
            )
        else:
            n_estimators_xgb = None
            max_depth_xbg = None
            learning_rate = None
            subsample = None
        params = {
            "kwargs": {
                "n_estimators": n_estimators_xgb,
                "max_depth": max_depth_xbg,
                "learning_rate": learning_rate,
                "subsample": subsample,
            }
        }
        st.divider()
    else:
        params = XGB_GRID

    model_types[ModelNames.XGBoost.value] = {
        "use": True,
        "params": params,
    }
    return model_types


def _svm_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        st.write("Options:")
        kernel = st.selectbox("Kernel", options=SVM_KERNELS)
        degree = st.number_input("Degree", min_value=0, value=3)
        c = st.number_input("C", value=1.0, min_value=0.0)
        params = {
            "kernel": kernel.lower(),
            "degree": degree,
            "C": c,
        }
        st.divider()
    else:
        params = SVM_GRID

    model_types[ModelNames.SVM.value] = {
        "use": True,
        "params": params,
    }
    return model_types


def _mlrem_opts(use_hyperparam_search: bool) -> dict:
    model_types = {}
    if not use_hyperparam_search:
        with st.expander("MLREM Options"):
            # Basic parameters
            alpha = st.number_input("Alpha (regularisation)", value=0.0, min_value=0.0, step=0.1)
            max_beta = st.number_input("Maximum Beta", value=40, min_value=1, step=1, 
                                     help="Will test beta values from 0.1 to max_beta * 0.1")
            weight_threshold = st.number_input(
                "Weight Threshold", value=1e-3, min_value=0.0, step=1e-4, format="%.4f",
                help="Features with weights below this will be removed")
            
            # Advanced options
            st.write("Advanced Options:")
            max_iterations = st.number_input("Max Iterations", value=300, min_value=1, step=50)
            tolerance = st.number_input("Tolerance", value=0.01, format="%.4f", step=0.001)
            
            # Feature selection options
            st.write("Feature Selection Options:")
            screen_variance = st.checkbox("Screen Low Variance Features", value=False)
            variance_threshold = st.number_input(
                "Variance Threshold", 
                value=0.01, 
                min_value=0.0, 
                step=0.01,
                help="Features with variance below this will be removed if screening is enabled"
            )
        
        params = {
            'alpha': alpha,
            'beta_range': [0.1 * i for i in range(1, max_beta + 1)],
            'weight_threshold': weight_threshold,
            'max_iterations': max_iterations,
            'tolerance': tolerance,
            'screen_variance': screen_variance,
            'variance_threshold': variance_threshold,
        }
    else:
        params = MLREM_GRID

    model_types[ModelNames.MLREM.value] = {
        "use": True,
        "params": params,
    }
    return model_types


@st.experimental_fragment
def preprocessing_opts_form(data: pd.DataFrame):
    st.write("## Data Preprocessing Options")

    st.write("### Data Normalisation")

    st.write(
        """
        If you select **"Standardisation"**, your data will be normalised by subtracting the
        mean and dividing by the standard deviation for each feature. The resulting transformation has a
        mean of 0 and values are between -1 and 1.

        If you select **"Minmax"**, your data will be scaled based on the minimum and maximum
        value of each feature. The resulting transformation will have values between 0 and 1.

        If you select **"None"**, the data will not be normalised.
        """
    )

    st.write("#### Normalisation Method for Independent Variables")

    st.selectbox(
        "Normalisation",
        NORMALISATIONS,
        key=DataPreprocessingStateKeys.IndependentNormalisation,
        index=len(NORMALISATIONS) - 1,  # default to no normalisation
    )

    st.write("#### Transformation Method for Dependent Variable")

    transformationy = st.selectbox(
        "Transformations",
        TRANSFORMATIONS_Y,
        key=DataPreprocessingStateKeys.DependentNormalisation,
        index=len(TRANSFORMATIONS_Y) - 1,  # default to no transformation
    )

    if (
        transformationy.lower() == TransformationsY.Log
        or transformationy.lower() == TransformationsY.Sqrt
    ):
        if (
            data.iloc[:, -1].min() <= 0
        ):  # deal with user attempting this transformations on negative values
            st.warning(
                "The dependent variable contains negative values. Log and square root transformations require positive values."
            )
            if st.checkbox(
                "Proceed with transformation. This option will add a constant to the dependent variable to make it positive.",
                key=DataPreprocessingStateKeys.ProceedTransformation,
            ):
                pass
            else:
                st.stop()

    st.write("### Feature Selection")

    st.write("#### Check the Feature Selection Algorithms to Use")

    variance_disabled = True
    if st.checkbox(
        "Variance threshold",
        key=DataPreprocessingStateKeys.VarianceThreshold,
        help="Delete features with variance below a certain threshold",
    ):
        variance_disabled = False
    st.number_input(
        "Set threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        key=DataPreprocessingStateKeys.ThresholdVariance,
        disabled=variance_disabled,
    )

    correlation_disabled = True
    if st.checkbox(
        "Correlation threshold",
        key=DataPreprocessingStateKeys.CorrelationThreshold,
        help="Delete features with correlation above a certain threshold",
    ):
        correlation_disabled = False
    st.number_input(
        "Set threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        key=DataPreprocessingStateKeys.ThresholdCorrelation,
        disabled=correlation_disabled,
    )

    lasso_disabled = True
    if st.checkbox(
        "Lasso Feature Selection",
        key=DataPreprocessingStateKeys.LassoFeatureSelection,
        help="Select features using Lasso regression",
    ):
        lasso_disabled = False
    st.number_input(
        "Set regularisation term",
        min_value=0.0,
        value=0.05,
        key=DataPreprocessingStateKeys.RegularisationTerm,
        disabled=lasso_disabled,
    )
