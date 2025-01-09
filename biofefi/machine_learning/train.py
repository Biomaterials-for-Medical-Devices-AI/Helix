import os

from biofefi.machine_learning.call_methods import save_actual_pred_plots
from biofefi.machine_learning.data import DataBuilder
from biofefi.machine_learning.learner import GridSearchLearner, Learner
from biofefi.options.execution import ExecutionOptions
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions
from biofefi.utils.logging_utils import Logger


def run(
    ml_opts: MachineLearningOptions,
    exec_opts: ExecutionOptions,
    plot_opts: PlottingOptions,
    data: DataBuilder,
    logger: Logger,
) -> None:
    """
    Run the ML training pipeline
    """

    if exec_opts.use_hyperparam_search:
        learner = GridSearchLearner(
            model_types=ml_opts.model_types,
            problem_type=exec_opts.problem_type,
            data_split=exec_opts.data_split,
            normalization=exec_opts.normalization,
            logger=logger,
        )
    else:
        learner = Learner(
            model_types=ml_opts.model_types,
            problem_type=exec_opts.problem_type,
            data_split=exec_opts.data_split,
            normalization=exec_opts.normalization,
            n_bootstraps=exec_opts.n_bootstraps,
            logger=logger,
        )
    res, metric_res, metric_res_stats, trained_models = learner.fit(data)
    logger.info(f"Performance Metric Statistics: {os.linesep}{metric_res_stats}")
    if ml_opts.save_actual_pred_plots:
        save_actual_pred_plots(
            data=data,
            ml_results=res,
            opt=exec_opts,
            logger=logger,
            ml_metric_results=metric_res,
            ml_metric_results_stats=metric_res_stats,
            ml_opts=ml_opts,
            plot_opts=plot_opts,
            n_bootstraps=exec_opts.n_bootstraps,
        )

    return trained_models, metric_res_stats
