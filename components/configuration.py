from options.enums import ConfigStateKeys
import streamlit as st


def ml_options():
    ml_on = True
    st.session_state[ConfigStateKeys.IsMachineLearning] = ml_on
    # ml_on = st.checkbox("Machine Learning", key=ConfigStateKeys.IsMachineLearning)
    if ml_on:
        with st.expander("Machine Learning Options"):
            st.subheader("Machine Learning Options")
            st.selectbox(
                "Problem type",
                ["Classification", "Regression"],
                key=ConfigStateKeys.ProblemType,
            ).lower()
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
            st.checkbox("Save actual or predicted plots", key=ConfigStateKeys.SavePlots)
            st.checkbox("Save models", key=ConfigStateKeys.SaveModels)

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
            st.session_state[ConfigStateKeys.ModelTypes] = model_types

            st.selectbox(
                "Normalization",
                ["Standardization", "MinMax", "None"],
                key=ConfigStateKeys.Normalization,
            )
