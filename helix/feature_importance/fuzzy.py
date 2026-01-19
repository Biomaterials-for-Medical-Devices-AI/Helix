from copy import deepcopy

import pandas as pd

from helix.options.enums import ProblemTypes
from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.fuzzy import FuzzyOptions
from helix.options.plotting import PlottingOptions
from helix.services.feature_importance.local_methods import (
    calculate_lime_values,
    calculate_local_shap_values,
)
from helix.services.feature_importance.results import (
    save_fuzzy_sets_plots,
    save_importance_results,
    save_target_clusters_plots,
)
from helix.utils.logging_utils import Logger


class Fuzzy:
    """
    Fuzzy class to interpret synergy of importance between features within context.

    """

    def __init__(
        self,
        fuzzy_opt: FuzzyOptions,
        fi_opt: FeatureImportanceOptions,
        exec_opt: ExecutionOptions,
        plot_opt: PlottingOptions,
        # ml_opt: argparse.Namespace,
        logger: Logger | None = None,
    ) -> None:
        self._fuzzy_opt = fuzzy_opt
        self._fi_opt = fi_opt
        self._exec_opt = exec_opt
        self._plot_opt = plot_opt
        self._logger = logger
        self._local_importance_methods = self._fi_opt.local_importance_methods
        self.importance_type = "local"  # local feature importance

    def interpret(self, models, ensemble_results, data):
        """
        Interpret the model results using the selected feature importance methods and ensemble methods.
        Parameters:
            models (dict): Dictionary of models.
            data (object): Data object.
        Returns:
            dict: Dictionary of feature importance results.
        """
        # create a copy of the data - select first fold of the data
        X_train, X_test = deepcopy(data.X_train[0]), deepcopy(data.X_test[0])
        self._logger.info("-------- Start of fuzzy interpretation logging--------")
        # Step 1: fuzzy feature selection to select top features for fuzzy interpretation
        if self._fuzzy_opt.fuzzy_feature_selection:
            # Select top features for fuzzy interpretation
            try:
                topfeatures = self._select_features(ensemble_results["Majority Vote"])
                X_train = X_train[topfeatures]
                X_test = X_test[topfeatures]
            except Exception as e:
                self._logger.error(f"Error in fuzzy feature selection: {e}")
                try:
                    # Use Mean ensemble results
                    topfeatures = self._select_features(ensemble_results["Mean"])
                    X_train = X_train[topfeatures]
                    X_test = X_test[topfeatures]
                except Exception as e:
                    self._logger.error(f"Error in fuzzy feature selection: {e}")

            # if error occurs, use Mean ensemble results

        # Step 2: Assign granularity to features e.g. low, medium, high categories
        if self._fuzzy_opt.granular_features:
            X_train = self._fuzzy_granularity(X_train)
            X_test = self._fuzzy_granularity(X_test)

        # Step 3: Master feature importance dataframe for granular features from local feature importance methods and ML models
        master_importance_df = self._local_feature_importance(
            models, data.X_train[0], data.y_train[0]
        )

        # Step 4: Extract fuzzy rules from master dataframe
        fuzzy_rules_df = self._fuzzy_rule_extraction(master_importance_df)
        save_importance_results(
            feature_importance_df=fuzzy_rules_df,
            model_type=None,
            importance_type="fuzzy",
            feature_importance_type="top contextual rules",
            experiment_name=self._exec_opt.experiment_name,
            fi_opt=self._fi_opt,
            plot_opt=self._plot_opt,
            logger=self._logger,
        )

        # Step 5: Identify the synergy of important features by context (e.g. target category:low, medium, high)
        df_contextual_rules = self._contextual_synergy_analysis(fuzzy_rules_df)
        save_importance_results(
            feature_importance_df=df_contextual_rules,
            model_type=None,
            importance_type="fuzzy",
            feature_importance_type="top contextual rules",
            experiment_name=self._exec_opt.experiment_name,
            fi_opt=self._fi_opt,
            plot_opt=self._plot_opt,
            logger=self._logger,
        )

        # local_importance_results = self._local_feature_importance(models, X, y)
        self._logger.info("-------- End of fuzzy interpretation logging--------")

        return df_contextual_rules

    def _select_features(self, ensemble_results):
        """
        Select top features from ensemble feature importance. This can be either
        majority vote-based importance or mean-based importance
        Parameters:
            ensemble_results: Dictionary of feature importance results.
        Returns:
            list: List of top features.
        """
        self._logger.info(
            f"Selecting top {self._fuzzy_opt.number_fuzzy_features} features..."
        )
        fi = ensemble_results.sort_values(by=0, ascending=False)
        # Select top n features for fuzzy interpretation
        topfeatures = fi.index[: self._fuzzy_opt.number_fuzzy_features].tolist()
        return topfeatures

    def _fuzzy_granularity(self, X):
        """
        Generate membership functions for each feature to split them into 3 granularities:
        "small", "moderate", "large". Then use these membership functions to determine
        degree membership for each granularity.

        The Z function is used to determine degree membership of the "small" granularity.

        The triangular function is used to determine degree membership of the "moderate"
        granularity.

        The S function is used to determine degree membership of the "large" granularity.

        Parameters:
            X (pd.DataFrame): Features.
        Returns:
            pd.DataFrame: Features with granularity.
        """
        import warnings

        import numpy as np
        import skfuzzy as fuzz

        # Suppress all warnings
        warnings.filterwarnings("ignore")
        self._logger.info("Assigning granularity to features...")
        # find interquartile values for each feature
        df_top_qtl = X.quantile([0, 0.25, 0.5, 0.75, 1])
        # Create membership functions based on interquartile values for each feature
        membership_functions = {}
        universe = {}
        for feature in X.columns:

            # Define the universe for each feature
            universe[feature] = np.linspace(X[feature].min(), X[feature].max(), 100)

            # Define membership functions
            # features with less than 3 unique values
            if len(X[feature].unique()) < 3:
                # print('Feature with only 2 values')
                # print(feature)
                # print(df_top_qtl[feature])
                low_mf = fuzz.trimf(
                    universe[feature],
                    [
                        df_top_qtl[feature][0.00],
                        df_top_qtl[feature][0.00],
                        df_top_qtl[feature][0.00],
                    ],
                )
                medium_mf = fuzz.trimf(
                    universe[feature],
                    [
                        (df_top_qtl[feature][0.00] + df_top_qtl[feature][1.00]) / 2,
                        (df_top_qtl[feature][0.00] + df_top_qtl[feature][1.00]) / 2,
                        (df_top_qtl[feature][0.00] + df_top_qtl[feature][1.00]) / 2,
                    ],
                )
                high_mf = fuzz.trimf(
                    universe[feature],
                    [
                        df_top_qtl[feature][1.00],
                        df_top_qtl[feature][1.00],
                        df_top_qtl[feature][1.00],
                    ],
                )

            # Highly skewed features
            elif df_top_qtl[feature][0.00] == df_top_qtl[feature][0.50]:
                low_mf = fuzz.trimf(
                    universe[feature],
                    [
                        df_top_qtl[feature][0.00],
                        df_top_qtl[feature][0.50],
                        df_top_qtl[feature][0.75],
                    ],
                )
                medium_mf = fuzz.trimf(
                    universe[feature],
                    [
                        df_top_qtl[feature][0.50],
                        df_top_qtl[feature][0.75],
                        df_top_qtl[feature][1.00],
                    ],
                )
                high_mf = fuzz.smf(
                    universe[feature],
                    df_top_qtl[feature][0.75],
                    df_top_qtl[feature][1.00],
                )

            else:
                low_mf = fuzz.zmf(
                    universe[feature],
                    df_top_qtl[feature][0.00],
                    df_top_qtl[feature][0.50],
                )
                medium_mf = fuzz.trimf(
                    universe[feature],
                    [
                        df_top_qtl[feature][0.25],
                        df_top_qtl[feature][0.50],
                        df_top_qtl[feature][0.75],
                    ],
                )
                high_mf = fuzz.smf(
                    universe[feature],
                    df_top_qtl[feature][0.50],
                    df_top_qtl[feature][1.00],
                )

            membership_functions[feature] = {
                "small": low_mf,
                "moderate": medium_mf,
                "large": high_mf,
            }
        if self._fuzzy_opt.save_fuzzy_set_plots:
            save_fuzzy_sets_plots(
                universe=universe,
                membership_functions=membership_functions,
                x_cols=X.columns,
                exec_opt=self._exec_opt,
                plot_opt=self._plot_opt,
                logger=self._logger,
            )

        # Create granular features using membership values
        new_df_features = []
        for feature in X.columns:
            X.loc[:, f"{feature}_small"] = fuzz.interp_membership(
                universe[feature], membership_functions[feature]["small"], X[feature]
            )
            new_df_features.append(f"{feature}_small")
            X.loc[:, f"{feature}_mod"] = fuzz.interp_membership(
                universe[feature], membership_functions[feature]["moderate"], X[feature]
            )
            new_df_features.append(f"{feature}_mod")
            X.loc[:, f"{feature}_large"] = fuzz.interp_membership(
                universe[feature], membership_functions[feature]["large"], X[feature]
            )
            new_df_features.append(f"{feature}_large")
        X = X[new_df_features]

        return X

    def _fuzzyset_selection(self, uni, mf1, mf2, mf3, val):
        """
        Select fuzzy set with highest membership value.
        Parameters:
            uni (np.array): Universe.
            mf1 (np.array): Low membership function
            mf2 (np.array): Moderate membership function
            mf3 (np.array): High membership function
            val (float): Value.
        Returns:
            str: Fuzzy set with highest membership value
        """
        import skfuzzy as fuzz

        mf_values = []
        # Calculate membership values for each fuzzy set
        mf_values.append(fuzz.interp_membership(uni, mf1, val))

        mf_values.append(fuzz.interp_membership(uni, mf2, val))

        mf_values.append(fuzz.interp_membership(uni, mf3, val))

        # Select fuzzy set with highest membership value
        # if multiple fuzzy sets have the same membership value, the first one is selected

        index_of_max = mf_values.index(max(mf_values))

        # Return fuzzy set
        match index_of_max:
            case 0:
                return "small"
            case 1:
                return "moderate"
            case 2:
                return "large"

    def _fuzzy_rule_extraction(self, df):
        """
        Cluster the local feature importance data into the user-defined clusters and then
        assign the "small", "moderate" and "large" granularities to features within those
        clusters.

        Parameters:
            df (Dataframe): master dataframe of feature importances from local feature importance methods and ML models.
        Returns:
            pd.DataFrame: Features with fuzzy rules.
        """
        import numpy as np
        import skfuzzy as fuzz

        self._logger.info("Extracting fuzzy rules...")
        if self._exec_opt.problem_type == ProblemTypes.Regression:
            target = np.array(df[df.columns[-1]])
            centers, membership_matrix, _, _, _, _, _ = fuzz.cluster.cmeans(
                data=target.reshape(1, -1),
                c=self._fuzzy_opt.number_clusters,
                m=2,  # Fuzziness parameter
                error=0.005,
                maxiter=1000,
            )
            # Determine the primary cluster assignment for each data point
            primary_cluster_assignment = np.argmax(membership_matrix, axis=0)

            # Calculate the average value for each cluster
            cluster_numbers = np.unique(primary_cluster_assignment).tolist()

            cluster_averages = [
                np.mean(target[primary_cluster_assignment == i])
                for i in range(self._fuzzy_opt.number_clusters)
            ]

            # Replace cluser numbers by cluster names based on their average values
            cluster_names = self._fuzzy_opt.cluster_names

            # Create a list of tuples where each tuple contains a cluster number and its corresponding average
            clusters = list(zip(cluster_numbers, cluster_averages))

            # Sort this list by the averages
            clusters.sort(key=lambda x: x[1])

            # Create a dictionary where the keys are the cluster numbers and the values are the cluster names
            cluster_mapping = {
                cluster_num[0]: cluster_name
                for cluster_num, cluster_name in zip(clusters, cluster_names)
            }

            # replace the cluster numbers in the list with the corresponding cluster names
            primary_cluster_assignment = [
                cluster_mapping[cluster_num]
                for cluster_num in primary_cluster_assignment
            ]

            # TODO: save the plot of the range of cluster assignments
            # create new dataframe with target values and cluster assignments
            df_cluster = pd.DataFrame(
                {
                    "target": df.loc[:, df.columns[-1]].to_list(),
                    "cluster": primary_cluster_assignment,
                }
            )

            if self._fuzzy_opt.save_fuzzy_set_plots:
                save_target_clusters_plots(
                    df_cluster=df_cluster,
                    exec_opt=self._exec_opt,
                    plot_opt=self._plot_opt,
                    logger=self._logger,
                )
            # Assign labels to target
            df.loc[:, df.columns[-1]] = primary_cluster_assignment

        # Create membership functions based on interquartile values for each feature
        membership_functions = {}
        universe = {}
        for feature in df.columns[:-1]:
            # Define the universe for each feature
            universe[feature] = np.linspace(df[feature].min(), df[feature].max(), 100)

            # Define membership functions
            low_mf = fuzz.zmf(universe[feature], 0.00, 0.5)
            medium_mf = fuzz.trimf(universe[feature], [0.25, 0.5, 0.75])
            high_mf = fuzz.smf(universe[feature], 0.5, 1.00)

            membership_functions[feature] = {
                "small": low_mf,
                "moderate": medium_mf,
                "large": high_mf,
            }

        # Create fuzzy rules
        fuzzy_rules = []

        # Loop through each row in the dataframe and extract fuzzy rules
        for i, _ in df.iterrows():
            df_instance = {}  # Dictionary to store observation values
            fuzzy_sets = {}  # Dictionary to store fuzzy sets
            for feature in df.columns[:-1]:
                df_instance[feature] = df.loc[i, feature]
            # Extract fuzzy set for each feature
            for feature in df.columns[:-1]:
                fuzzy_sets[feature] = self._fuzzyset_selection(
                    universe[feature],
                    membership_functions[feature]["small"],
                    membership_functions[feature]["moderate"],
                    membership_functions[feature]["large"],
                    df_instance[feature],
                )

            fuzzy_sets[df.columns[-1]] = df.loc[i, df.columns[-1]]
            fuzzy_rules.append(fuzzy_sets)

        # Create dataframe of fuzzy rules
        fuzzy_rules_df = pd.DataFrame(fuzzy_rules, index=df.index)

        return fuzzy_rules_df

    def _contextual_synergy_analysis(self, fuzzy_rules_df):
        """
        Identify most occuring fuzzy rules by context (e.g. target category:low, medium, high)
        Parameters:
            fuzzy_rules_df (pd.DataFrame): Features with fuzzy rules.
        Returns:
            pd.DataFrame: Most occuring fuzzy rules.
        """
        self._logger.info(
            "Use most occuring fuzzy rules to extract synergy of features..."
        )
        # Drop rules with all NaN values
        fuzzy_rules_df.dropna(how="all", axis=1, inplace=True)

        # Group by all columns to count the number of occurences of each rule
        fuzzy_rules_grouped = (
            fuzzy_rules_df.groupby(fuzzy_rules_df.columns.tolist())
            .size()
            .reset_index()
            .rename(columns={0: "records"})
        )

        # Sort by the number of occurences
        fuzzy_rules_grouped = fuzzy_rules_grouped.sort_values(
            by=["records"], ascending=False
        )

        # Identify top n rules in each target category
        top_rules = {}
        for category in fuzzy_rules_grouped[fuzzy_rules_grouped.columns[-2]].unique():
            top_rules[category] = fuzzy_rules_grouped[
                fuzzy_rules_grouped[fuzzy_rules_grouped.columns[-2]] == category
            ].head(self._fuzzy_opt.number_rules)

        synergy_features = {}
        for category, rules in top_rules.items():
            synergy_features[category] = {}
            for feature in rules.columns[:-2]:
                unique_values = rules[feature].unique()
                if "large" in unique_values:
                    top_value = "large"
                elif "moderate" in unique_values:
                    top_value = "moderate"
                else:
                    top_value = "small"
                synergy_features[category][feature] = top_value

        self._logger.info(f"synergy and impact of features: \n{synergy_features}")
        # Convert dictionary to dataframe
        synergy_features_df = pd.DataFrame(synergy_features)
        return synergy_features_df

    def _local_feature_importance(self, models, X, y):
        """
        Calculate feature importance for a given model and dataset.
        Parameters:
            models (dict): Dictionary of models.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
        Returns:
            dict: Dictionary of feature importance results.
        """
        self._logger.info("Creating master feature importance dataframe...")
        feature_importance_results = {}

        if not any(self._local_importance_methods.values()):
            self._logger.info("No local feature importance methods selected")
        else:
            for model_type, model in models.items():
                self._logger.info(
                    f"Local feature importance methods for {model_type}..."
                )

                feature_importance_results[model_type] = {}

                # Run methods with TRUE values in the dictionary of feature importance methods
                for (
                    feature_importance_type,
                    value,
                ) in self._local_importance_methods.items():
                    # Select the first model in the list - model[0]
                    if value["value"]:
                        if feature_importance_type == "LIME":
                            # Run LIME importance
                            lime_importance_df = calculate_lime_values(
                                model=model[0],
                                X=X,
                                problem_type=self._exec_opt.problem_type,
                                logger=self._logger,
                            )
                            # Normalise LIME coefficients between 0 and 1 (0 being the lowest impact and 1 being the highest impact)
                            lime_importance_df = lime_importance_df.abs()
                            lime_importance_df_norm = (
                                lime_importance_df - lime_importance_df.min()
                            ) / (lime_importance_df.max() - lime_importance_df.min())
                            # Add class to local feature importance
                            lime_importance_df_norm = pd.concat(
                                [lime_importance_df_norm, y], axis=1
                            )
                            # lime_importance_df = pd.concat([lime_importance_df, y], axis=1)
                            feature_importance_results[model_type][
                                feature_importance_type
                            ] = lime_importance_df_norm

                        if feature_importance_type == "SHAP":
                            # Run SHAP
                            shap_df, shap_values = calculate_local_shap_values(
                                model=model[0],
                                X=X,
                                logger=self._logger,
                            )
                            # Normalise SHAP values between 0 and 1 (0 being the lowest impact and 1 being the highest impact)
                            shap_df = shap_df.abs()
                            shap_df_norm = (shap_df - shap_df.min()) / (
                                shap_df.max() - shap_df.min()
                            )
                            # Add class to local feature importance
                            shap_df_norm = pd.concat([shap_df_norm, y], axis=1)
                            feature_importance_results[model_type][
                                feature_importance_type
                            ] = shap_df_norm

        # Concatenate the results
        master_df = pd.DataFrame()
        for model_type, feature_importance in feature_importance_results.items():
            for feature_importance_type, result in feature_importance.items():
                master_df = pd.concat([master_df, result], axis=0)

        # Reset the index of the master dataframe
        master_df.reset_index(drop=True, inplace=True)

        return master_df
