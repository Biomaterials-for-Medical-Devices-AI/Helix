import logging
import time
from pathlib import Path


class Logger(object):
    """
    logger preparation


    Parameters
    ----------
    log_dir: Path
        path to the log directory

    logging_level: str
        required Level of logging. INFO, WARNING or ERROR can be selected.
        Default to 'INFO'

    console_logger: bool
        flag if console_logger is required. Default to False

    Returns
    ----------
    logger: logging.Logger
        logger object
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        logging_level: str = "INFO",
        console_logger: bool = True,
        multi_module: bool = True,
    ) -> None:
        super().__init__()
        self._log_dir = log_dir
        self.console_logger = console_logger
        self.logging_level = logging_level.lower()
        self.multi_module = multi_module
        self._make_level()

    def _make_level(self):
        if self.logging_level == "info":
            self._level = logging.INFO
        elif self.logging_level == "warning":
            self._level = logging.WARNING
        elif self.logging_level == "error":
            self._level = logging.ERROR
        else:
            raise ValueError(
                "logging_level not specified correctly. "
                "INFO, WARNING or ERROR must be chosen"
            )

    def make_logger(self):
        # Create a custom logger
        if self.multi_module:
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(__name__)
        logger.setLevel(self._level)

        # Create formatter
        format = logging.Formatter("%(levelname)s - %(message)s - %(module)s")

        # Create file handler
        if self._log_dir is not None:
            # logging configuration
            self._log_dir.mkdir(parents=True, exist_ok=True)
            file_name = self._log_dir.joinpath(f"{time.strftime('%Y%m%d-%H%M%S')}.log")
            f_handler = logging.FileHandler(filename=file_name)
            f_handler.setLevel(self._level)
            f_handler.setFormatter(format)
            logger.addHandler(f_handler)

        # Console handler creation
        if self.console_logger:
            c_handler = logging.StreamHandler()
            c_handler.setLevel(self._level)
            c_handler.setFormatter(format)
            logger.addHandler(c_handler)

        return logger

    def remove_handlers(self, logger):
        """
        Remove all handlers associated with the logger object.
        """
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def close_logger(logger_instance, logger):
    logger_instance.remove_handlers(logger)
    del logger_instance
    del logger
    return None
