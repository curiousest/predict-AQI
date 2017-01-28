from sklearn.neural_network import MLPRegressor

from multi_city_model import (
    load_and_clean_locations_near_airlocation,
    generate_AQI_inputs_and_outputs,
    generate_predictions_two_step,
    generate_baseline_predictions,
    cut_off_end_split_function
)
MAX_INDEX = 48


def test_hyperparams_on_airlocation(
        airlocation_id,
        indices_ahead_to_predict_range,
        list_of_indices_behind_to_use_range,
        list_of_number_of_nearby_locations_to_use,
        list_of_hidden_layer_sizes=[(100, 10)],
        list_of_alphas=[0.0001]):

    # load, align, and clean the data from db
    max_locations_needed = max(list_of_number_of_nearby_locations_to_use)
    df, airlocation_ids, continuous_time_series = load_and_clean_locations_near_airlocation(
        airlocation_id, 50, max_locations_needed
    )

    # generate inputs and target outputs for the model
    df, continuous_time_series, input_columns, output_columns = generate_AQI_inputs_and_outputs(
        df,
        continuous_time_series,
        indices_ahead_to_predict_range,
        max(list_of_indices_behind_to_use_range),
        max_locations_needed
    )

    city_specific_input_columns = filter(lambda col: 'loc' in col, input_columns)
    other_input_columns = list(set(input_columns) ^ set(city_specific_input_columns))

    row_count = df.count()[0]
    split_row = int(round(row_count * 0.90))

    # Make baseline predictions
    baseline_df = df.copy(deep=True)
    baseline_df = generate_baseline_predictions(baseline_df, output_columns)

    for indices_behind_to_use_range in list_of_indices_behind_to_use_range:
        for number_of_nearby_locations_to_use in list_of_number_of_nearby_locations_to_use:
            for hidden_layer_sizes in list_of_hidden_layer_sizes:
                for alpha in list_of_alphas:
                    locations = ['loc_{}'.format(str(i)) for i in range(1, number_of_nearby_locations_to_use + 1)]
                    regressor_params = {'alpha': alpha, 'hidden_layer_sizes': hidden_layer_sizes}
                    first_step_regressors = [MLPRegressor(**regressor_params) for i in locations]
                    second_step_regressors = [MLPRegressor(**regressor_params) for i in indices_ahead_to_predict_range]
                    this_iteration_df = df.copy(deep=True)
                    this_iteration_df = generate_predictions_two_step(
                        this_iteration_df,
                        other_input_columns,
                        output_columns,
                        cut_off_end_split_function,
                        locations,
                        first_step_regressors,
                        second_step_regressors,
                        indices_ahead_to_predict_range,
                        indices_behind_to_use_range,
                        "ahead_single_city_pred",
                        False
                    )

                    # calculate errors...

