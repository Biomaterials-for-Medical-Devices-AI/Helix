# Create a BaseOptions class to define the basic options used in the project.
# This class is the parent class of MLOptions, FIOptions and FEOptions.

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BaseOptions:
    """
    Base options - parent class for all options
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self) -> None:
        self.parser.add_argument(
            "--experiment_name",
            type=str,
            default="test",
            help="Name of the experiment",
        ),
        self.parser.add_argument(
            "--is_feature_engineering",
            type=bool,
            default=False,
            help="Flag for feature engineering",
        ),
        self.parser.add_argument(
            "--is_machine_learning",
            type=bool,
            default=False,
            help="Flag for machine learning",
        ),
        self.parser.add_argument(
            "--is_feature_importance",
            type=bool,
            default=False,
            help="Flag for feature importance",
        ),

                
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self._opt = self.parser.parse_args()

        return self._opt