# 2. Test page loads without error
# 3. a) Test manual Linear Model
# 3. b) Test manual Random Forest
# 3. c) Test manual SVM
# 3. d) Test manual XGBoost
# 4. a) Test AHPS Linear Model
# 4. b) Test AHPS Random Forest
# 4. c) Test AHPS SVM
# 4. d) Test AHPS XGBoost


import pytest
from .fixtures import (
    execution_opts,
    plotting_opts,
    data_opts,
    dummy_data,
    new_experiment,
)
