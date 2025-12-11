import os

from helix.machine_learning.learner import GridSearchLearner, Learner
from helix.options.data import DataOptions
from helix.options.execution import ExecutionOptions
from helix.options.ml import MachineLearningOptions
from helix.options.plotting import PlottingOptions
from helix.services.data import DataBuilder
from helix.services.machine_learning.results import save_actual_pred_plots
from helix.utils.logging_utils import Logger


def run(
    ml_opts: MachineLearningOptions,
    data_opts: DataOptions,
    plot_opts: PlottingOptions,
    data: DataBuilder,
    exec_opts: ExecutionOptions,
    logger: Logger,
    n_cpus: int,
) -> None:
    """
    Run the ML training pipeline
    """

    if ml_opts.use_hyperparam_search:
        learner = GridSearchLearner(
            model_types=ml_opts.model_types,
            problem_type=exec_opts.problem_type,
            data_split=data_opts.data_split,
            logger=logger,
            n_cpus=n_cpus,
        )
    else:
        learner = Learner(
            model_types=ml_opts.model_types,
            problem_type=exec_opts.problem_type,
            data_split=data_opts.data_split,
            logger=logger,
            n_cpus=n_cpus,
        )
    res, metrics_full, metrics_mean_std, trained_models = learner.fit(data)
    logger.info(f"Performance Metric Statistics: {os.linesep}{metrics_mean_std}")
    if ml_opts.save_actual_pred_plots:
        save_actual_pred_plots(
            data=data,
            ml_results=res,
            logger=logger,
            ml_metric_results=metrics_full,
            ml_metric_results_stats=metrics_mean_std,
            exec_opts=exec_opts,
            plot_opts=plot_opts,
            ml_opts=ml_opts,
            trained_models=trained_models,
        )

    return trained_models, metrics_full, metrics_mean_std
