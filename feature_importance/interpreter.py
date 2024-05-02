import argparse

from call_methods import save_importance_results
from ensemble_methods import calculate_ensemble_mean, calculate_ensemble_majorityvote
from feature_importance_methods import calculate_permutation_importance, calculate_shap_values

class Interpreter:
    """
    Interpreter class to interpret the model results.

    """

    def __init__(self, opt: argparse.Namespace) -> None:
        self._opt = opt
        self._feature_importance_methods = self._opt.feature_importance_methods
        self._feature_importance_ensemble= self._opt.feature_importance_ensemble

    
    def interpret(self, models, X, y):
        '''
        Interpret the model results using the selected feature importance methods and ensemble methods.
        Parameters:
            models (dict): Dictionary of models.
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
        Returns:
            dict: Dictionary of feature importance results.
        '''
        feature_importance_results = self._individual_feature_importance(models, X, y)
        ensemble_results = self._ensemble_feature_importance(feature_importance_results)

        return feature_importance_results, ensemble_results
         



    def _individual_feature_importance(self, models,  X, y):
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

        if not any(self._feature_importance_methods.values()):
            print("No feature importance methods selected")
        else:
            for model_type, model in models.items():
                print(f"Calculating feature importance for {model_type}...")

                # Run methods with TRUE values in the dictionary of feature importance methods
                for feature_importance_type, value in self._feature_importance_methods.items():
                    if value:
                        if feature_importance_type == 'Permutation Importance':
                            # Run Permutation Importance
                            print(f"Calculating {feature_importance_type} importance...")
                            permutation_importance_df = calculate_permutation_importance(model, X, y, self._opt)
                            save_importance_results(permutation_importance_df, model_type, feature_importance_type, self._opt)
                            feature_importance_results[feature_importance_type] = permutation_importance_df

                        if feature_importance_type == 'SHAP':
                            # Run SHAP
                            print(f"Calculating {feature_importance_type} importance...")
                            shap_df, shap_values = calculate_shap_values(model, X,self._opt)
                            save_importance_results(shap_df, model_type, feature_importance_type, self._opt, shap_values)
                            feature_importance_results[feature_importance_type] = shap_df

        return feature_importance_results
    
    def _ensemble_feature_importance(self, feature_importance_results):
        '''
        Calculate ensemble feature importance methods.
        Parameters:
            feature_importance_results (dict): Dictionary of feature importance results.
        Returns:
            dict: Dictionary of ensemble feature importance results.
        '''
        ensemble_results = {}

        if not any(self._feature_importance_ensemble.values()):
            print("No ensemble feature importance method selected")
        else:            
            print("------ Calculating ensemble of feature importance results ------")
            for ensemble_type, value in self._feature_importance_ensemble.items():
                if value:
                    if ensemble_type == 'Mean':
                        # Calculate mean of feature importance results            
                        mean_results = calculate_ensemble_mean(feature_importance_results, self._opt)
                        save_importance_results(mean_results, None, ensemble_type, self._opt)
                        ensemble_results[ensemble_type] = mean_results
                    
                    if ensemble_type == 'Majority Vote':
                        # Calculate majority vote of feature importance results            
                        majority_vote_results = calculate_ensemble_majorityvote(feature_importance_results, self._opt)
                        save_importance_results(majority_vote_results, None, ensemble_type, self._opt)
                        ensemble_results[ensemble_type] = majority_vote_results
        
        return ensemble_results




    