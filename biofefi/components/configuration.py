import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from biofefi.options.choices import (
    DATA_SPLITS,
    NORMALISATIONS,
    PLOT_FONT_FAMILIES,
    PROBLEM_TYPES,
    SVM_KERNELS,
)
from biofefi.options.enums import ConfigStateKeys, DataSplitMethods, PlotOptionKeys


@st.experimental_fragment
def ml_options_box():
    ml_on = st.checkbox(
        "Train new models", key=ConfigStateKeys.IsMachineLearning, value=True
    )
    with st.expander("Machine Learning Options"):
        if ml_on:
            st.subheader("Machine Learning Options")
            st.selectbox(
                "Problem type",
                PROBLEM_TYPES,
                key=ConfigStateKeys.ProblemType,
            )

            st.write("Model types to use:")
            model_types = {}
            use_linear = st.checkbox("Linear Model", value=True)
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

            use_rf = st.checkbox("Random Forest", value=True)
            if use_rf:
                st.write("Options:")
                n_estimators_rf = st.number_input(
                    "Number of estimators", value=300, key="n_estimators_rf"
                )
                min_samples_split = st.number_input("Minimum samples split", value=2)
                min_samples_leaf = st.number_input("Minimum samples leaf", value=1)
                max_depth_rf = st.number_input(
                    "Maximum depth", value=6, key="max_depth_rf"
                )
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

            use_xgb = st.checkbox("XGBoost", value=True)
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

            use_svm = st.checkbox("SVM", value=True)
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

        st.selectbox(
            "Normalization",
            NORMALISATIONS,
            key=ConfigStateKeys.Normalization,
        )

        data_split = st.selectbox("Data split method", ["Holdout", "K-Fold"])
        if data_split == "Holdout":
            split_size = st.number_input(
                "Test split",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
            )
            st.session_state[ConfigStateKeys.DataSplit] = {
                "type": "holdout",
                "test_size": split_size,
            }
        elif data_split == "K-Fold":
            split_size = st.number_input(
                "n splits",
                min_value=0,
                value=5,
            )
            st.session_state[ConfigStateKeys.DataSplit] = {
                "type": "kfold",
                "n_splits": split_size,
            }
        else:
            split_size = None
        st.number_input(
            "Number of bootstraps",
            min_value=1,
            value=10,
            key=ConfigStateKeys.NumberOfBootstraps,
        )
        st.checkbox("Save models", key=ConfigStateKeys.SaveModels)


@st.experimental_fragment
def plot_options_box():
    """Expander containing the options for making plots"""
    with st.expander("Plot options", expanded=False):
        save = st.checkbox(
            "Save all plots",
            key=PlotOptionKeys.SavePlots,
            value=True,
        )
        rotate_x = st.number_input(
            "Angle to rotate X-axis labels",
            min_value=0,
            max_value=90,
            value=10,
            key=PlotOptionKeys.RotateXAxisLabels,
            disabled=not save,
        )
        rotate_y = st.number_input(
            "Angle to rotate Y-axis labels",
            min_value=0,
            max_value=90,
            value=60,
            key=PlotOptionKeys.RotateYAxisLabels,
            disabled=not save,
        )
        tfs = st.number_input(
            "Title font size",
            min_value=20,
            key=PlotOptionKeys.TitleFontSize,
            disabled=not save,
        )
        afs = st.number_input(
            "Axis font size",
            min_value=8,
            key=PlotOptionKeys.AxisFontSize,
            disabled=not save,
        )
        ats = st.number_input(
            "Axis tick size",
            min_value=8,
            key=PlotOptionKeys.AxisTickSize,
            disabled=not save,
        )
        cs = st.selectbox(
            "Colour scheme",
            options=plt.style.available,
            key=PlotOptionKeys.ColourScheme,
            disabled=not save,
        )
        font = st.selectbox(
            "Font",
            options=PLOT_FONT_FAMILIES,
            key=PlotOptionKeys.FontFamily,
            disabled=not save,
            index=1,
        )
        if save:
            st.write("### Preview")
            plt.style.use(cs)
            arr = np.random.normal(1, 0.5, size=100)
            data = pd.DataFrame({"A": arr, "B": arr, "C": arr})
            fig, ax = plt.subplots()
            sns.violinplot(data=data, ax=ax)
            ax.set_title("Title", fontsize=tfs, family=font)
            ax.set_xlabel("X axis", fontsize=afs, family=font)
            ax.set_ylabel("Y axis", fontsize=afs, family=font)
            ax.tick_params(labelsize=ats)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_x, family=font)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=rotate_y, family=font)
            st.pyplot(fig, clear_figure=True)
            fig.clear()


@st.experimental_fragment
def fi_options_box():
    fi_on = st.checkbox("Feature Importance", key=ConfigStateKeys.IsFeatureImportance)
    if fi_on:
        with st.expander("Feature importance options"):
            st.write("Global feature importance methods:")
            global_methods = {}
            use_permutation = st.checkbox("Permutation Importance")
            global_methods["Permutation Importance"] = {
                "type": "global",
                "value": use_permutation,
            }
            use_shap = st.checkbox("SHAP")
            global_methods["SHAP"] = {"type": "global", "value": use_shap}
            st.session_state[ConfigStateKeys.GlobalFeatureImportanceMethods] = (
                global_methods
            )

            st.write("Feature importance ensemble methods:")
            ensemble_methods = {}
            use_mean = st.checkbox("Mean")
            ensemble_methods["Mean"] = use_mean
            use_majority = st.checkbox("Majority vote")
            ensemble_methods["Majority Vote"] = use_majority
            st.session_state[ConfigStateKeys.EnsembleMethods] = ensemble_methods

            st.write("Local feature importance methods:")
            local_importance_methods = {}
            use_lime = st.checkbox("LIME")
            local_importance_methods["LIME"] = {"type": "local", "value": use_lime}
            use_local_shap = st.checkbox("Local SHAP")
            local_importance_methods["SHAP"] = {
                "type": "local",
                "value": use_local_shap,
            }
            st.session_state[ConfigStateKeys.LocalImportanceFeatures] = (
                local_importance_methods
            )

            st.number_input(
                "Number of most important features to plot",
                min_value=1,
                value=10,
                key=ConfigStateKeys.NumberOfImportantFeatures,
            )
            st.selectbox(
                "Scoring function for permutation importance",
                [
                    "neg_mean_absolute_error",
                    "neg_root_mean_squared_error",
                    "accuracy",
                    "f1",
                ],
                key=ConfigStateKeys.ScoringFunction,
            )
            st.number_input(
                "Number of repetitions for permutation importance",
                min_value=1,
                value=5,
                key=ConfigStateKeys.NumberOfRepetitions,
            )
            st.slider(
                "Percentage of data to consider for SHAP",
                0,
                100,
                100,
                key=ConfigStateKeys.ShapDataPercentage,
            )
            st.checkbox(
                "Save feature importance options",
                key=ConfigStateKeys.SaveFeatureImportanceOptions,
            )
            st.checkbox(
                "Save feature importance results",
                key=ConfigStateKeys.SaveFeatureImportanceResults,
            )

            # Fuzzy Options
            st.subheader("Fuzzy Options")
            fuzzy_feature_selection = st.checkbox(
                "Fuzzy feature selection", key=ConfigStateKeys.FuzzyFeatureSelection
            )
            if fuzzy_feature_selection:
                st.number_input(
                    "Number of features for fuzzy interpretation",
                    min_value=1,
                    value=5,
                    key=ConfigStateKeys.NumberOfFuzzyFeatures,
                )
                st.checkbox("Granular features", key=ConfigStateKeys.GranularFeatures)
                st.number_input(
                    "Number of clusters for target variable",
                    min_value=2,
                    value=3,
                    key=ConfigStateKeys.NumberOfClusters,
                )
                st.text_input(
                    "Names of clusters (comma-separated)",
                    key=ConfigStateKeys.ClusterNames,
                )
                st.number_input(
                    "Number of top occurring rules for fuzzy synergy analysis",
                    min_value=1,
                    value=10,
                    key=ConfigStateKeys.NumberOfTopRules,
                )


@st.experimental_fragment
def execution_options_box():
    st.write(
        """
        If your dependent variable is categorical (e.g. cat 🐱 or dog 🐶), choose **"Classification"**.

        If your dependent variable is continuous (e.g. stock prices 📈), choose **"Regression"**.
        """
    )
    st.selectbox(
        "Problem type",
        PROBLEM_TYPES,
        key=ConfigStateKeys.ProblemType,
    )
    st.write(
        """
        If you select **"Standardization"**, your data will be normalised by subtracting the
        mean and dividing by the standard deviation for each feature. The resulting transformation has a
        mean of 0 and values are between -1 and 1.

        If you select **"Minmax"**, your data will be scaled based on the minimum and maximum
        value of each feature. The resulting transformation will have values between 0 and 1.

        If you select **"None"**, the data will not be normalised.
        """
    )
    st.selectbox(
        "Normalisation",
        NORMALISATIONS,
        key=ConfigStateKeys.Normalization,
    )
    data_split = st.selectbox("Data split method", DATA_SPLITS)
    if data_split.lower() == DataSplitMethods.Holdout:
        split_size = st.number_input(
            "Test split",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
        )
        st.session_state[ConfigStateKeys.DataSplit] = {
            "type": DataSplitMethods.Holdout,
            "test_size": split_size,
        }
    elif data_split.lower() == DataSplitMethods.KFold:
        split_size = st.number_input(
            "n splits",
            min_value=0,
            value=5,
        )
        st.session_state[ConfigStateKeys.DataSplit] = {
            "type": DataSplitMethods.KFold,
            "n_splits": split_size,
        }
    else:
        split_size = None
    st.number_input(
        "Number of bootstraps",
        min_value=1,
        value=10,
        key=ConfigStateKeys.NumberOfBootstraps,
    )
    st.number_input(
        "Random seed", value=1221, min_value=0, key=ConfigStateKeys.RandomSeed
    )
