import math
import itertools
import pandas as pd

from predict_aqi import config

from predict_aqi.transform_data import (
    generate_time_inputs, shift_and_save_column, get_normalized_aqi
)
from predict_aqi.predictor_utils import (
    train_regressor, predict_values, print_mean_absolute_error
)


def generate_predictions():
    pass


def generate_AQI_inputs_and_outputs():
    pass
