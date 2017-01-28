import datetime
import itertools
import pandas as pd

from predict_aqi import config

from predict_aqi.load_data import load_nearby_airlocations_to_list_of_dataframes
from predict_aqi.transform_data import (
    generate_time_inputs, shift_and_save_column, get_normalized_aqi,
    clean_data, align_multi_location_time_series_data
)
from predict_aqi.predictor_utils import (
    train_regressor, predict_values
)


def load_and_clean_locations_near_airlocation(
        airlocation_id,
        distance_km,
        max_locations=3,
        print_progress=False):
    # load from db to multiple data frames
    df_list, airlocation_ids = load_nearby_airlocations_to_list_of_dataframes(
        airlocation_id, distance_km, max_locations, print_progress=print_progress
    )

    # merge and align to a single data frame
    start = datetime.datetime.now()
    df, continuous_time_series = align_multi_location_time_series_data(df_list, len(airlocation_ids))
    end = datetime.datetime.now()

    if print_progress:
        print("Aligning data took {} seconds".format((end - start).total_seconds()))
    return df, airlocation_ids, continuous_time_series


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

    # remove null columns due to the column shifting
    df = clean_data(df, output_columns, input_columns, remove_dirty=True)

    return df, continuous_time_series, input_columns, output_columns





# returns the data frame up to the split row
# iloc uses row index after filtering as if a re-indexing occurred (as opposed to using the indexing
# that existed before the dirty data filtering happened)
def cut_off_end_split_function(all_data, input_columns, output_columns, cut_off_percentage=0.9):
    # Split by the first 90% of data. That means our testing set is extrapolation.
    row_count = all_data.count()[0]
    split_row = int(round(row_count * cut_off_percentage))
    # x_train, y_train, x_test, y_test
    return (all_data[input_columns].iloc[:split_row],
            all_data[output_columns].iloc[:split_row],
            all_data[input_columns].iloc[split_row:],
            all_data[output_columns].iloc[split_row:])


def generate_predictions_two_step(all_data,
                                  additional_second_step_features,
                                  output_columns,
                                  split_function,
                                  first_step_locations,
                                  first_step_regressors,
                                  second_step_regressors,
                                  indices_ahead_to_predict,
                                  indices_behind_to_use,
                                  prediction_column_suffix,
                                  print_progress=True,
                                  ):
    if print_progress:
        print("Step 1")

    # make a regressor + predictions for each location
    for location, regressor in zip(first_step_locations, first_step_regressors):
        # the feature columns depend on the city, but they are all
        # trying to predict the AQI of the target city
        feature_columns = ["{}_ago_{}".format(i, location)
                           for i in indices_behind_to_use]
        # print(feature_columns)
        x_train, y_train, x_test, y_test = split_function(
            all_data, feature_columns, output_columns
        )
        train_regressor(regressor, x_train, y_train, print_progress)

        # Make the first step predictions and merge them into the all_data dataframe
        first_step_predictions = predict_values(regressor, all_data[feature_columns], print_progress)
        first_step_predictions_df = pd.DataFrame(first_step_predictions, index=all_data.index)
        prediction_columns = ['{}_ahead_{}_first_step_pred'.format(str(i), location)
                              for i in indices_ahead_to_predict]
        first_step_predictions_df.columns = prediction_columns
        all_data = all_data.join(first_step_predictions_df)

    if print_progress:
        print("Step 2")

    # make a regressor + predictions for each "x hours ahead" we want to predict
    for index_ahead_to_predict, output_column, regressor in zip(indices_ahead_to_predict, output_columns,
                                                                second_step_regressors):
        # Use only the prediction from the first step that corresponds to this step's regressor, otherwise
        # it will overfit using all the first step prediction values.
        second_step_input_columns_from_first = ['{}_ahead_{}_first_step_pred'.format(index_ahead_to_predict, location)
                                                for location in first_step_locations]
        feature_columns = list(itertools.chain(
            additional_second_step_features, second_step_input_columns_from_first
        ))

        x_train, y_train, x_test, y_test = split_function(
            all_data, feature_columns, [output_column]
        )
        train_regressor(regressor, x_train, y_train, print_progress)

        # Make the second step predictions and merge them into the all_data dataframe
        second_step_predictions = predict_values(
            regressor, all_data[feature_columns], print_progress
        )
        second_step_predictions_df = pd.DataFrame(second_step_predictions, index=all_data.index)
        prediction_columns = ['{}_{}'.format(str(index_ahead_to_predict), prediction_column_suffix)]
        second_step_predictions_df.columns = prediction_columns
        all_data = all_data.join(second_step_predictions_df)

    return all_data


def generate_baseline_predictions(df, output_columns, source_column='loc_1_aqi'):
    for output_column in output_columns:
        prediction_column_name = output_column.replace('_ahead_AQI', '_ahead_baseline_pred')
        df[prediction_column_name] = df[source_column]
    return df

