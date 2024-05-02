import os
import sys

from machine_learning.data import DataBuilder
from machine_learning.learner import Learner
from machine_learning.ml_options import MLOptions
from utils.logging_utils import Logger
from utils.utils import set_seed


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
    res, metric_res, trained_models = learner.fit(data)
    logger.info(f"Results: \n{metric_res}")
    return trained_models
