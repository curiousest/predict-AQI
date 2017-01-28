import math
import itertools
import pandas as pd

from predict_aqi import config

from predict_aqi.transform_data import (
    generate_time_inputs, shift_and_save_column, get_normalized_aqi, clean_data
)
from predict_aqi.predictor_utils import (
    train_regressor, predict_values, print_mean_absolute_error
)


def generate_predictions():
    pass


def generate_AQI_inputs_and_outputs(df,
                                    continuous_time_series,
                                    indices_ahead_to_predict,
                                    indices_behind_to_use,
                                    number_of_locations,
                                    output_column_format="{}_ahead_AQI",
                                    source_column_format="loc_{}_aqi",
                                    normalized_column_format="loc_{}_normalized_AQI"):

    for loc_number in range(1, number_of_locations + 1):
        df[normalized_column_format.format(loc_number)] = \
            df[source_column_format.format(loc_number)].map(get_normalized_aqi)

    df, output_columns = generate_outputs(
        df,
        indices_ahead_to_predict,
        "loc_1_normalized_AQI",
        output_column_format
    )

    # We have to cut out the rows where a value in x_ahead_AQI is incorrect
    # (because there was no data or there was a time shift ahead)
    for index, (start, end) in enumerate(continuous_time_series):
        continuous_time_series[index] = (start, end - indices_ahead_to_predict[-1])

    df, time_columns = generate_time_inputs(df, time_column="loc_1_measurement_datetime")

    input_columns = time_columns

    # generate the x_ago_loc_y columns
    for loc_number in range(1, number_of_locations + 1):
        input_column_format = '{}' + "_ago_loc_{}".format(loc_number)
        for index in indices_behind_to_use:
            input_column = input_column_format.format(str(index))
            input_columns.append(input_column)
            shift_and_save_column(df, "loc_{}_normalized_AQI".format(loc_number), input_column, shift=index)

    return df, continuous_time_series, input_columns, output_columns


def generate_outputs(df,
                     indices_ahead_to_predict,
                     AQI_column,
                     output_column_format="{}_ahead_AQI"):
    '''
    Returns a dataframe with the future AQI values that are supposed to be predicted for each row.
    '''
    output_columns = []

    for index in indices_ahead_to_predict:
        output_column = output_column_format.format(str(index))
        output_columns.append(output_column)
        shift_and_save_column(df, AQI_column, output_column, shift=-index)
    return df, output_columns
