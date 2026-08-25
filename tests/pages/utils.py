from streamlit.testing.v1 import AppTest

from helix.options.enums import ViewExperimentKeys
from tests.utils import get_element_by_key


def select_experiment(at: AppTest, experiment: str):
    """In the provided AppTest instance, select an experiment to be used
    in the test pages.

    Args:
        at (AppTest): The AppTest instance.
        experiment (str): The name of the experiment to select.
    """
    exp_selector = get_element_by_key(
        at, "selectbox", ViewExperimentKeys.ExperimentName
    )
    exp_selector.select(experiment).run()
