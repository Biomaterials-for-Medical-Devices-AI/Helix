import argparse

from feature_importance.call_methods import save_importance_results
from feature_importance.feature_importance_methods import (
    calculate_shap_values, calculate_lime_values)

class Fuzzy:
    """
    Fuzzy class to interpret synergy of importance between features within context.

    """

    def __init__(self, opt: argparse.Namespace, logger: object = None) -> None:
        self._opt = opt
        self._logger = logger
        self._local_importance_methods = self._opt.local_importance_methods
        self.importance_type = 'local' # local feature importance


    
    def interpret(self, models, ensemble_results, data):
        '''
        Interpret the model results using the selected feature importance methods and ensemble methods.
        Parameters:
            models (dict): Dictionary of models.
            data (object): Data object.
        Returns:
            dict: Dictionary of feature importance results.
        '''
        # create a copy of the data
        X_train, X_test, _, _ = data.X_train, data.X_test, data.y_train, data.y_test
        self._logger.info(f"-------- Start of fuzzy interpretation logging--------")
        if self._opt.fuzzy_feature_selection:
            # Select top features for fuzzy interpretation
            topfeatures = self._select_features(ensemble_results['Majority Vote'])
            X_train = X_train[topfeatures]
            X_test = X_test[topfeatures]
        if self._opt.is_granularity:
            X_train = self._fuzzy_granularity(X_train)
            X_test = self._fuzzy_granularity(X_test)
        #local_importance_results = self._local_feature_importance(models, X, y)
        self._logger.info(f"-------- End of fuzzy interpretation logging--------") 

        return X_train, X_test

    def _select_features(self, majority_vote_results):
        '''
        Select top features from majority vote ensemble feature importance.
        Parameters:
            majority_vote_results: Dictionary of feature importance results.
        Returns:
            list: List of top features.
        '''
        self._logger.info(f"Selecting top {self._opt.number_fuzzy_features} features...")
        fi = majority_vote_results.sort_values(by=0, ascending=False)
        # Select top n features for fuzzy interpretation
        topfeatures = fi.index[:self._opt.number_fuzzy_features].tolist()
        return topfeatures
    
    def _fuzzy_granularity(self, X):
        '''
        Assign granularity to features.
        Parameters:
            X (pd.DataFrame): Features.
        Returns:
            pd.DataFrame: Features with granularity.
        '''
        import numpy as np
        import skfuzzy as fuzz
        import warnings

        # Suppress all warnings
        warnings.filterwarnings('ignore'
                                )
        self._logger.info(f"Assigning granularity to features...")
        # find interquartile values for each feature
        df_top_qtl = X.quantile([0,0.25, 0.5, 0.75,1])
        # Create membership functions based on interquartile values for each feature
        membership_functions = {}
        universe = {}
        for feature in X.columns:
            
            # Define the universe for each feature
            universe[feature] = np.linspace(X[feature].min(), X[feature].max(), 100)

            # Define membership functions
            # Highly skewed features
            if df_top_qtl[feature][0.00] == df_top_qtl[feature][0.50]:
                low_mf = fuzz.trimf(universe[feature], [df_top_qtl[feature][0.00],df_top_qtl[feature][0.50],
                                                        df_top_qtl[feature][0.75]])
                medium_mf = fuzz.trimf(universe[feature], [df_top_qtl[feature][0.50],df_top_qtl[feature][0.75],
                                                        df_top_qtl[feature][1.00]])
                high_mf = fuzz.smf(universe[feature], df_top_qtl[feature][0.75], df_top_qtl[feature][1.00])
            
            else:
                low_mf = fuzz.zmf(universe[feature], df_top_qtl[feature][0.00],df_top_qtl[feature][0.50])
                medium_mf = fuzz.trimf(universe[feature], [df_top_qtl[feature][0.25],df_top_qtl[feature][0.50],
                                                        df_top_qtl[feature][0.75]])
                high_mf = fuzz.smf(universe[feature], df_top_qtl[feature][0.50], df_top_qtl[feature][1.00])
            
            membership_functions[feature] = {'low': low_mf, 'medium': medium_mf, 'high': high_mf}

        # Create granular features using membership values
        new_df_features = []
        for feature in X.columns:
            X.loc[:, f'{feature}_small'] = fuzz.interp_membership(universe[feature], membership_functions[feature]['low'], X[feature])
            new_df_features.append(f'{feature}_small')
            X.loc[:, f'{feature}_mod'] = fuzz.interp_membership(universe[feature], membership_functions[feature]['medium'], X[feature])
            new_df_features.append(f'{feature}_mod')
            X.loc[:, f'{feature}_large'] = fuzz.interp_membership(universe[feature], membership_functions[feature]['high'], X[feature])
            new_df_features.append(f'{feature}_large')
        X = X[new_df_features]
                
        return X

    
    def _local_feature_importance(self, models,  X, y):
        '''
        Calculate feature importance for a given model and dataset.
        Parameters:
            models (dict): Dictionary of models.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
        Returns:
            dict: Dictionary of feature importance results.
        '''
        feature_importance_results = {}

        if not any(self._local_importance_methods.values()):
            self._logger.info("No local feature importance methods selected")
        else:
            for model_type, model in models.items():
                self._logger.info(f"Local feature importance methods for {model_type}...")

                # Run methods with TRUE values in the dictionary of feature importance methods
                for feature_importance_type, value in self._local_importance_methods.items():
                    if value['value']:
                        if feature_importance_type == 'LIME':
                            # Run Permutation Importance                            
                            lime_importance_df = calculate_lime_values(model, X, self._opt,self._logger)
                            save_importance_results(lime_importance_df, model_type, self.importance_type,
                                                    feature_importance_type, self._opt,self._logger)
                            feature_importance_results[feature_importance_type] = lime_importance_df

                        if feature_importance_type == 'SHAP':
                            # Run SHAP
                            shap_df, shap_values = calculate_shap_values(model, X, value['type'], self._opt,self._logger)
                            save_importance_results(shap_df, model_type,self.importance_type, 
                                                    feature_importance_type, self._opt, self._logger,shap_values)
                            feature_importance_results[feature_importance_type] = shap_df

        return feature_importance_results




    
         
