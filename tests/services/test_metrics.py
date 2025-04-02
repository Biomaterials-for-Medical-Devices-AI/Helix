from helix.services.metrics import find_mean_model_index
from helix.options.enums import Metrics

def test_find_mean_model_index_normal_case():
    # Arrange
    metrics_dict = {
        'linear model': {
            'test': {
                'R2': {
                    'mean': -2.282950310005943,
                    'std': 0.2573510871565634
                }
            }
        }
    }
    expected_index = 2  # Index of 0.73, closest to mean of 0.75

    # Act
    actual_index = find_mean_model_index(metrics_dict, Metrics.R2.value)

    # Assert
    assert actual_index == expected_index

def test_find_mean_model_index_different_metric():
    # Arrange
    metrics_dict = {
        'linear model': {
            'test': {
                'MAE': {
                    'mean': 17.12743902155728,
                    'std': 0.29570889977778503
                }
            }
        }
    }
    expected_index = 0  # Index of 0.09, closest to mean of 0.1

    # Act
    actual_index = find_mean_model_index(metrics_dict, Metrics.MSE.value)

    # Assert
    assert actual_index == expected_index

def test_find_mean_model_index_empty_bootstraps():
    # Arrange
    metrics_dict = {
        "test": {Metrics.R2.value: {"mean": 0.8}},
        "bootstraps": []
    }
    expected_index = -1  # Should return -1 for empty bootstraps

    # Act
    actual_index = find_mean_model_index(metrics_dict, Metrics.R2.value)

    # Assert
    assert actual_index == expected_index