# 1. Test no exceptin occurs loading the page
# 2. Test an experiment is created at the given directory
# 3. Check the execution options exist and have the expected values
# 4. Check the plotting options exist and have the expected values
# 5. Check the data has been uploaded to the experiment

from streamlit.testing.v1 import AppTest


def test_page_loads_without_error():
    # Arrange
    at = AppTest.from_file("helix/pages/1_New_Experiment.py")

    # Act
    at.run()

    # Assert
    assert not at.exception
    assert not at.error
