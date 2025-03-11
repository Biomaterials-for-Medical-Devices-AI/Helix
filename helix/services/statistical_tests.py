import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Union, Optional


def kolmogorov_smirnov_test(
    data: Union[np.ndarray, list],
    reference_dist: Optional[str] = 'norm'
) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test to determine if a sample comes from a reference distribution.
    By default, tests against a normal distribution.

    Args:
        data: Input array of observations to test.
              Can be a numpy array or a list.
        reference_dist: String specifying the reference distribution.
                       Default is 'norm' for normal distribution.
                       Other options include: 'uniform', 'expon', etc.

    Returns:
        Tuple containing:
            - statistic: The test statistic
            - p_value: The p-value for the hypothesis test

    Note:
        - Null hypothesis: the data comes from the specified distribution
        - If p-value < alpha (typically 0.05), reject the null hypothesis 
          (data does not come from the specified distribution)
        - If p-value >= alpha, fail to reject the null hypothesis 
          (data may come from the specified distribution)
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if len(data.shape) > 1:
        data = data.flatten()
    
    # Fit the reference distribution to the data
    if reference_dist == 'norm':
        # For normal distribution, we need mean and std
        params = stats.norm.fit(data)
        return stats.kstest(data, reference_dist, args=params)
    else:
        # For other distributions, let scipy handle the parameter fitting
        return stats.kstest(data, reference_dist)


def shapiro_wilk_test(data: Union[np.ndarray, list]) -> Tuple[float, float]:
    """
    Perform Shapiro-Wilk test for normality on the input data.

    The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a
    normal distribution.

    Args:
        data: Input array of observations to test for normality.
              Can be a numpy array or a list.

    Returns:
        Tuple containing:
            - statistic: The test statistic
            - p_value: The p-value for the hypothesis test

    Note:
        - Null hypothesis: the data is normally distributed
        - If p-value < alpha (typically 0.05), reject the null hypothesis 
          (data is not normally distributed)
        - If p-value >= alpha, fail to reject the null hypothesis 
          (data may be normally distributed)
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if len(data.shape) > 1:
        data = data.flatten()
    
    return stats.shapiro(data)


def create_normality_test_table(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Create a dataframe with normality test results for numerical columns.

    Args:
        data: Input DataFrame containing the data to test

    Returns:
        DataFrame containing normality test results for each numerical column,
        or None if no valid columns are found
    """
    test_results = []
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        # Skip if all values are the same (no variance)
        if len(data[col].unique()) <= 1:
            continue
            
        # Perform Shapiro-Wilk test
        sw_stat, sw_p = shapiro_wilk_test(data[col].dropna())
        
        # Perform Kolmogorov-Smirnov test
        ks_stat, ks_p = kolmogorov_smirnov_test(data[col].dropna())
        
        test_results.append({
            'Variable': col,
            'Shapiro-Wilk Statistic': round(sw_stat, 3),
            'Shapiro-Wilk p-value': round(sw_p, 3),
            'Kolmogorov-Smirnov Statistic': round(ks_stat, 3),
            'Kolmogorov-Smirnov p-value': round(ks_p, 3)
        })
    
    return pd.DataFrame(test_results) if test_results else None
