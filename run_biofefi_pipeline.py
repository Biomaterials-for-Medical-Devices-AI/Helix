from feature_importance import feature_importance, fuzzy_interpretation
from feature_importance.feature_importance_options import FeatureImportanceOptions
from feature_importance.fuzzy_options import FuzzyOptions
from machine_learning import train
from machine_learning.data import DataBuilder
from machine_learning.ml_options import MLOptions
from utils.logging_utils import Logger, close_logger
from utils.utils import set_seed

fuzzy_opt = FuzzyOptions().parse()
fi_opt = FeatureImportanceOptions().parse()
ml_opt = MLOptions().parse()
seed = ml_opt.random_state

# Set up logger
logger_instance = Logger(ml_opt.ml_log_dir)
logger = logger_instance.make_logger()

# Set seed for reproducibility
set_seed(seed)

# Data ingestion
data = DataBuilder(ml_opt, logger).ingest()

# Machine learning
trained_models = train.run(ml_opt, data, logger)

# Feature importance
fi_logger_instance = Logger(fi_opt.fi_log_dir)
fi_logger = fi_logger_instance.make_logger()
gloabl_importance_results, local_importance_results, ensemble_results = (
    feature_importance.run(fi_opt, data, trained_models, fi_logger)
)

# Fuzzy interpretation
fuzzy_logger_instance = Logger(fuzzy_opt.fuzzy_log_dir)
fuzzy_logger = fuzzy_logger_instance.make_logger()
fuzzy_rules = fuzzy_interpretation.run(
    fuzzy_opt, ml_opt, data, trained_models, ensemble_results, fuzzy_logger
)

# Close the logger
close_logger(logger_instance, logger)
