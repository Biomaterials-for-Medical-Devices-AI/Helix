import os

from biofefi.machine_learning.data import DataBuilder
from biofefi.machine_learning.learner import GridSearchLearner, Learner
from biofefi.options.data import DataOptions
from biofefi.options.enums import ProblemTypes
from biofefi.options.execution import ExecutionOptions
from biofefi.options.ml import MachineLearningOptions
from biofefi.options.plotting import PlottingOptions
from biofefi.services.machine_learning.results import save_actual_pred_plots
from biofefi.utils.logging_utils import Logger


def run(
    ml_opts: MachineLearningOptions,
    data_opts: DataOptions,
    plot_opts: PlottingOptions,
    data: DataBuilder,
    problem_type: ProblemTypes,
    logger: Logger,
) -> None:
    """
    Run the ML training pipeline
    """

    if ml_opts.use_hyperparam_search:
        learner = GridSearchLearner(
            model_types=ml_opts.model_types,
            problem_type=problem_type,
            data_split=data_opts.data_split,
            logger=logger,
        )
    else:
        learner = Learner(
            model_types=ml_opts.model_types,
            problem_type=problem_type,
            data_split=data_opts.data_split,
            n_bootstraps=data_opts.data_split.n_bootstraps,
            logger=logger,
        )
    res, metric_res, metric_res_stats, trained_models = learner.fit(data)
    logger.info(f"Performance Metric Statistics: {os.linesep}{metric_res_stats}")
    if ml_opts.save_actual_pred_plots:
        if data_opts.data_split.n_bootstraps is not None:
            n = data_opts.data_split.n_bootstraps
        elif data_opts.data_split.k_folds is not None:
            n = data_opts.data_split.n_bootstraps
        else:
            n = 1
        save_actual_pred_plots(
            data=data,
            ml_results=res,
            opt=data_opts,
            logger=logger,
            ml_metric_results=metric_res,
            ml_metric_results_stats=metric_res_stats,
            ml_opts=ml_opts,
            plot_opts=plot_opts,
            n_bootstraps=n,
            trained_models=trained_models,
        )

    return trained_models, metric_res_stats
