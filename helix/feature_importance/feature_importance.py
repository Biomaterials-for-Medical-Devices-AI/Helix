from pathlib import Path

from helix.feature_importance.interpreter import FeatureImportanceEstimator
from helix.options.data import DataOptions
from helix.options.execution import ExecutionOptions
from helix.options.fi import FeatureImportanceOptions
from helix.options.plotting import PlottingOptions
from helix.services.data import TabularData


def run(
    fi_opt: FeatureImportanceOptions,
    exec_opt: ExecutionOptions,
    plot_opt: PlottingOptions,
    data_opt: DataOptions,
    data: TabularData,
    models,
    data_path: Path,
    logger,
):

    # Interpret the model results
    interpreter = FeatureImportanceEstimator(
        fi_opt=fi_opt,
        exec_opt=exec_opt,
        plot_opt=plot_opt,
        data_opt=data_opt,
        logger=logger,
        data_path=data_path,
    )
    gloabl_importance_results, local_importance_results, ensemble_results = (
        interpreter.interpret(models, data)
    )

    return gloabl_importance_results, local_importance_results, ensemble_results
