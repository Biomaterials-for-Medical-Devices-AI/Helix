import pandas as pd
import pytest

from biofefi.services.preprocessing import find_non_numeric_columns


def test_find_non_numeric_columns():
    """
    Test the find_non_numeric_columns function with both DataFrame and Series inputs.
    """

    # Arrange
    df_mixed = pd.DataFrame(
        {
            "A": ["1", "2", "x"],  # Contains a non-numeric value
            "B": ["4", "5", "6"],  # All numeric
            "C": [1.2, 3.4, 5.6],  # All numeric
        }
    )

    df_all_numeric = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.1, 6.2]})

    series_numeric = pd.Series([1, 2, 3], name="NumericSeries")
    series_non_numeric = pd.Series(["1", "2", "x"], name="NonNumericSeries")

    other_type = [1, 2, 3]

    # Act
    result_df_mixed = find_non_numeric_columns(df_mixed)
    result_df_all_numeric = find_non_numeric_columns(df_all_numeric)
    result_series_numeric = find_non_numeric_columns(series_numeric)
    result_series_non_numeric = find_non_numeric_columns(series_non_numeric)

    # Assert
    assert result_df_mixed == ["A"], "Failed: Should detect column 'A' as non-numeric"
    assert (
        result_df_all_numeric == []
    ), "Failed: Should return empty list for all numeric DataFrame"
    assert (
        result_series_numeric == []
    ), "Failed: Should return None for fully numeric Series"
    assert (
        result_series_non_numeric == "NonNumericSeries"
    ), "Failed: Should return the series name for non-numeric values"

    with pytest.raises(TypeError, match="Input must be a pandas DataFrame or Series."):
        find_non_numeric_columns(other_type)
