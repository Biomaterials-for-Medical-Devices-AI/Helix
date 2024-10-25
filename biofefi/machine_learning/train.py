from biofefi.machine_learning.data import DataBuilder
from biofefi.machine_learning.learner import Learner
from biofefi.machine_learning.ml_options import MLOptions
from biofefi.utils.logging_utils import Logger
from biofefi.machine_learning.call_methods import save_actual_pred_plots


def run(opt: MLOptions, data: DataBuilder, logger: Logger) -> None:
    """
    Run the ML training pipeline
    """

    # opt = MLOptions().parse()
    # seed = opt.random_state
    # logger = Logger(opt.log_dir, opt.experiment_name).make_logger()

    # # Set seed for reproducibility
    # set_seed(seed)
    learner = Learner(opt, logger=logger)
    res, metric_res, metric_res_stats, trained_models = learner.fit(data)
    logger.info(f"Performance Metric Statistics: \n{metric_res_stats}")
    if opt.save_actual_pred_plots:
        save_actual_pred_plots(data, res, opt, logger, metric_res)

    return trained_models
