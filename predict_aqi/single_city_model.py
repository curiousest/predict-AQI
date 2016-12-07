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


def generate_predictions(all_data,
                         first_step_format_inputs_outputs_function,
                         first_step_split_function,
                         second_step_format_inputs_outputs_function,
                         second_step_split_function,
                         first_step_regressor,
                         second_step_regressor,
                         indices_ahead_to_predict,
                         print_progress=True
                         ):

    if print_progress:
        print("Step 1")
    first_step_data, first_step_feature_columns, first_step_output_columns = first_step_format_inputs_outputs_function(
        all_data
    )
    x_train, y_train, x_test, y_test = first_step_split_function(
        first_step_data, first_step_feature_columns, first_step_output_columns
    )
    train_regressor(first_step_regressor, x_train, y_train, print_progress)

    # Make the first step predictions and merge them into the all_data dataframe
    first_step_predictions = predict_values(first_step_regressor, all_data[first_step_feature_columns], print_progress)
    first_step_predictions_df = pd.DataFrame(first_step_predictions, index=all_data.index)
    prediction_columns = ['{}_ahead_first_step_pred'.format(str(i)) for i in indices_ahead_to_predict]
    first_step_predictions_df.columns = prediction_columns
    all_data = all_data.join(first_step_predictions_df)

    if print_progress:
        print("Step 2")
    # Second step of model is to make predictions combining recent AQI predictions with time inputs
    second_step_data, second_step_feature_columns, second_step_output_columns = \
        second_step_format_inputs_outputs_function(all_data)
    x_train, y_train, x_test, y_test = second_step_split_function(
        second_step_data, second_step_feature_columns, second_step_output_columns
    )
    train_regressor(second_step_regressor, x_train, y_train, print_progress)

    # Make the second step predictions and merge them into the all_data dataframe
    second_step_predictions = predict_values(
        second_step_regressor, all_data[second_step_feature_columns], print_progress
    )
    second_step_predictions_df = pd.DataFrame(second_step_predictions, index=all_data.index)
    prediction_columns = ['{}_ahead_second_step_pred'.format(str(i)) for i in indices_ahead_to_predict]
    second_step_predictions_df.columns = prediction_columns
    all_data = all_data.join(second_step_predictions_df)

    return all_data


def generate_AQI_inputs_and_outputs(measurements,
                                    indices_behind_to_use,
                                    indices_ahead_to_predict,
                                    source_column_name=None,
                                    output_column_format="{}_ahead_AQI",
                                    input_column_format="{}_ago_AQI",
                                    ):
    all_data = measurements
    if source_column_name is None:
        source_column_name = "normalized_AQI"
        all_data[source_column_name] = all_data['aqi'].map(get_normalized_aqi)

    all_data, output_columns = generate_outputs(
        measurements, indices_ahead_to_predict, source_column_name, output_column_format
    )
    all_data, input_columns = generate_inputs_for_recent_AQI(
        all_data, indices_behind_to_use, source_column_name, input_column_format
    )

    # We have to cut out the rows where there is a None input - so the first and last few columns that have None
    # inputs due to shifting.
    all_data = all_data[indices_behind_to_use[-1] + 1: -indices_ahead_to_predict[-1]]

    return all_data, input_columns, output_columns


def generate_outputs(measurements, indices_ahead_to_predict, AQI_column, output_column_format="{}_ahead_AQI"):
    '''
    Returns a dataframe with the future AQI values that are supposed to be predicted for each row.
    '''
    output_columns = []

    for index in indices_ahead_to_predict:
        output_column = output_column_format.format(str(index))
        output_columns.append(output_column)
        shift_and_save_column(measurements, AQI_column, output_column, shift=-index)
    return measurements, output_columns


def generate_inputs_for_recent_AQI(measurements, indices_behind_to_use, AQI_column, input_column_format="{}_ago_AQI"):
    '''
    Returns a dataframe with the recent AQI inputs for each row.
    '''
    input_columns = []

    for index in indices_behind_to_use:
        input_column = input_column_format.format(str(index))
        input_columns.append(input_column)
        shift_and_save_column(measurements, AQI_column, input_column, shift=index)
    return measurements, input_columns


def cut_off_end_split_function(all_data, input_columns, output_columns, cut_off_percentage=0.9):
    # Split by the first 90% of data. That means we're our testing set is extrapolation.
    row_count = all_data.count()[0]
    split_row = int(round(row_count * cut_off_percentage))
    # x_train, y_train, x_test, y_test
    return (all_data[input_columns][:split_row],
            all_data[output_columns][:split_row],
            all_data[input_columns][split_row:],
            all_data[output_columns][split_row:])


def get_first_step_functions(input_columns, output_columns):

    def first_step_format_inputs_outputs(all_data):
        return all_data, input_columns, output_columns

    return first_step_format_inputs_outputs, cut_off_end_split_function


def get_second_step_functions(input_columns, output_columns, indices_ahead_to_predict):

    def second_step_format_inputs_outputs(all_data):
        all_data, time_columns = generate_time_inputs(all_data)
        first_step_prediction_columns = ['{}_ahead_first_step_pred'.format(str(i))
                                         for i in indices_ahead_to_predict]
        second_input_columns = list(itertools.chain(time_columns, first_step_prediction_columns))
        # return the right columns
        return all_data, second_input_columns, output_columns

    return second_step_format_inputs_outputs, cut_off_end_split_function
