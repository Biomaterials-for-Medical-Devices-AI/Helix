import pytest

from helix.options.enums import Metrics
from helix.services.metrics import find_mean_model_index


@pytest.fixture
def full_metrics():
    # Arrange
    return {
        "random forest": [
            {
                "MAE": {
                    "train": {"value": 1.3812835882049548},
                    "test": {"value": 4.068466435241056},
                },
                "RMSE": {
                    "train": {"value": 2.139398429871655},
                    "test": {"value": 6.247665398896417},
                },
                "R2": {
                    "train": {"value": 0.9594063312717612},
                    "test": {"value": 0.6943262096854719},
                },
            },
            {
                "MAE": {
                    "train": {"value": 1.3967932013646212},
                    "test": {"value": 3.821645833452862},
                },
                "RMSE": {
                    "train": {"value": 2.1796423137257483},
                    "test": {"value": 5.990756940840749},
                },
                "R2": {
                    "train": {"value": 0.9585034388755299},
                    "test": {"value": 0.7043514165281184},
                },
            },
            {
                "MAE": {
                    "train": {"value": 1.3967934909005144},
                    "test": {"value": 4.1158194443920015},
                },
                "RMSE": {
                    "train": {"value": 2.181018704876805},
                    "test": {"value": 6.430755346504488},
                },
                "R2": {
                    "train": {"value": 0.9604500544935055},
                    "test": {"value": 0.5754442869954024},
                },
            },
            {
                "MAE": {
                    "train": {"value": 1.4094596942557542},
                    "test": {"value": 3.7740717594655835},
                },
                "RMSE": {
                    "train": {"value": 2.217748694600388},
                    "test": {"value": 5.894237850517626},
                },
                "R2": {
                    "train": {"value": 0.9584062867330696},
                    "test": {"value": 0.6731658629254342},
                },
            },
            {
                "MAE": {
                    "train": {"value": 1.3804968728932452},
                    "test": {"value": 3.9481909723651945},
                },
                "RMSE": {
                    "train": {"value": 2.1165562144308305},
                    "test": {"value": 6.1732715852692},
                },
                "R2": {
                    "train": {"value": 0.9622937744051184},
                    "test": {"value": 0.6339172262415971},
                },
            },
            {
                "MAE": {
                    "train": {"value": 1.437824299321487},
                    "test": {"value": 3.8463645836497222},
                },
                "RMSE": {
                    "train": {"value": 2.217027529003114},
                    "test": {"value": 6.085433632755594},
                },
                "R2": {
                    "train": {"value": 0.956613554264147},
                    "test": {"value": 0.7063100165076914},
                },
            },
            {
                "MAE": {
                    "train": {"value": 1.42597810983959},
                    "test": {"value": 3.8180439815060563},
                },
                "RMSE": {
                    "train": {"value": 2.2414253799441286},
                    "test": {"value": 5.772024050547077},
                },
                "R2": {
                    "train": {"value": 0.9565766063993179},
                    "test": {"value": 0.7141467274356106},
                },
            },
            {
                "MAE": {
                    "train": {"value": 1.4529519921683733},
                    "test": {"value": 3.7617997686599174},
                },
                "RMSE": {
                    "train": {"value": 2.331340430955068},
                    "test": {"value": 5.497355612316541},
                },
                "R2": {
                    "train": {"value": 0.9547700864655906},
                    "test": {"value": 0.693759620431769},
                },
            },
            {
                "MAE": {
                    "train": {"value": 1.4232125898215213},
                    "test": {"value": 3.716966435252972},
                },
                "RMSE": {
                    "train": {"value": 2.2115601817923976},
                    "test": {"value": 5.623615179177445},
                },
                "R2": {
                    "train": {"value": 0.9576300638863458},
                    "test": {"value": 0.7309537204358652},
                },
            },
            {
                "MAE": {
                    "train": {"value": 1.405799166133815},
                    "test": {"value": 3.9340844909116943},
                },
                "RMSE": {
                    "train": {"value": 2.2202921914883222},
                    "test": {"value": 6.114012457883356},
                },
                "R2": {
                    "train": {"value": 0.956393092372356},
                    "test": {"value": 0.7052158172858851},
                },
            },
        ]
    }


@pytest.fixture
def aggregated_metrics():
    # Arrange
    return {
        "random forest": {
            "train": {
                "MAE": {"mean": 1.4110593004903875, "std": 0.02258219160363063},
                "RMSE": {"mean": 2.2056010070688457, "std": 0.0558687412196628},
                "R2": {"mean": 0.9581043289166742, "std": 0.002093474029074743},
            },
            "test": {
                "MAE": {"mean": 3.880545370489706, "std": 0.12603559117375532},
                "RMSE": {"mean": 5.98291280547085, "std": 0.274198905605784},
                "R2": {"mean": 0.6831590904472844, "std": 0.04368102565225929},
            },
        }
    }


@pytest.mark.parametrize(
    ["metric", "expected_index"],
    [(Metrics.R2.value, 3), (Metrics.MAE.value, 5)],
)
def test_find_mean_model_index_normal_case(
    full_metrics, aggregated_metrics, metric, expected_index
):
    # Act
    actual_index = find_mean_model_index(full_metrics, aggregated_metrics, metric)

    # Assert
    assert actual_index == expected_index
