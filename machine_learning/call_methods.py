import argparse
import os


def save_actual_pred_plots(data, results, opt: argparse.Namespace, logger) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    """Save Actual vs Predicted plots for Regression models
    Args:
        data: Data object
        results: Results of the model
        opt: Options
        logger: Logger
    Returns:
        None
    """
    if opt.problem_type == "regression":

        # Create results directory if it doesn't exist
        directory = opt.ml_log_dir
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        # Convert y_test to numpy arrays for easier handling
        y_test = [np.array(df) for df in data.y_test]

        # Plotting
        for model_name, model_options in opt.model_types.items():
            if model_options["use"]:
                logger.info(f"Saving actual vs prediction plot of {model_name}...")
                plt.figure(figsize=(10, 6))

                for i in range(opt.n_bootstraps):
                    y_pred_test = results[i][model_name]["y_pred_test"]

                    sns.scatterplot(
                        x=y_test[i], y=y_pred_test, marker="o", s=30, color="black"
                    )

                plt.title(f"Test Sets - {model_name}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.savefig(f"{directory}/{model_name}.png")
                # plt.show()
                plt.close()
