import datetime
import math
import itertools
import numpy as np
from operator import itemgetter

from predict_aqi import config


def make_circular_input(input, input_max, input_min=0):
    '''
    Convert the input to a circular input between 0-1.
    Circular input being two values which cause the smallest and
    largest values inputted to this function appear to be beside each other.

    This is used, for example, to make the last minute of the day "close" to
    the first minute of the day.
    '''
    # normalize the input to a real 0-1, then project it along
    # the first period of the sine/cosine waves
    normalized_projected = 2 * math.pi * (input - input_min) / (input_max - input_min)

    # apply sine / cosine and convert to a real number 0-1
    sine = (1 + math.sin(normalized_projected)) / 2
    cosine = (1 + math.cos(normalized_projected)) / 2
    assert 1 >= sine >= 0
    assert 1 >= cosine >= 0
    return sine, cosine


def get_normalized_time_inputs(dt):
    try:
        return tuple(itertools.chain(
            # minute of day
            make_circular_input((dt.time().hour * 60) + dt.time().minute, 1440),
            # day of the year
            make_circular_input(dt.timetuple().tm_yday, 365),
            # day of the week, from 0 to 6
            make_circular_input(dt.date().weekday(), 6),
            # day of the month
            make_circular_input(dt.date().day, 31, 1),
        ))
    except ValueError:
        return None, None, None, None, None, None, None, None


def generate_time_inputs(df, time_column="measurement_datetime"):
    df['minute_of_day_sin'], df['minute_of_day_cos'], \
    df['day_of_year_sin'], df['day_of_year_cos'], \
    df['day_of_week_sin'], df['day_of_week_cos'], \
    df['day_of_month_sin'], df['day_of_month_cos'] = \
        zip(*df[time_column].map(get_normalized_time_inputs))
    time_columns = ['minute_of_day_sin', 'minute_of_day_cos', 'day_of_year_cos', 'day_of_year_sin', 'day_of_week_cos',
               'day_of_week_sin', 'day_of_month_sin', 'day_of_month_cos']
    return df, time_columns


def get_normalized_aqi(aqi):
    if not aqi:
        return 0
    return min(aqi, config.MAX_AQI) / 300


def get_denormalized_aqi(normalized_aqi):
    try:
        if normalized_aqi is None or normalized_aqi <= 0:
            return 0
        return round(normalized_aqi * 300)
    except Exception:
        return 0


def generate_baseline_predictions(df, output_columns, source_column='aqi'):
    for output_column in output_columns:
        prediction_column_name = output_column.replace('_ahead_AQI', '_ahead_baseline_pred')
        df[prediction_column_name] = df[source_column]
    return df


def shift_and_save_column(df, source_col_name, dest_col_name, shift=1):
    df[dest_col_name] = getattr(df, source_col_name).shift(shift)


def shift_inputs_backwards(x_all, shift_count, source_column_name, column_string_to_format='{}_ago'):
    '''
    Returns x_all dataframe with `shift_count` new columns shifting the source column backwards one row for each new
    column and returns the column names.
    '''
    feature_columns = []
    shift_and_save_column(x_all, source_column_name, column_string_to_format.format('0'))
    feature_columns.append(column_string_to_format.format('0'))

    for i in range(1, shift_count):
        prev_feature_column = column_string_to_format.format(str(i))
        feature_column = column_string_to_format.format(str(i + 1))
        feature_columns.append(feature_column)
        shift_and_save_column(x_all, prev_feature_column, feature_column)
    return x_all, feature_columns


def shift_outputs_forwards(y_all, shift_count, source_column_name, column_string_to_format='{}_ahead'):
    shift_and_save_column(y_all, source_column_name, column_string_to_format.format('1'), shift=-1)
    output_columns = [column_string_to_format.format('1')]
    for i in range(2, shift_count + 1):
        prev_input_column = column_string_to_format.format(str(i - 1))
        output_column = column_string_to_format.format(str(i))
        output_columns.append(output_column)
        shift_and_save_column(y_all, prev_input_column, output_column, shift=-1)
    return y_all, output_columns


def too_small_or_empty(values, after_x_in_a_row):
    return len(values) < after_x_in_a_row or any(map(np.isnan, values))


def clean_data(df, input_columns, after_x_in_a_row=3, remove_dirty=True):
    '''
    Removes a row
    '''
    df['is_dirty'] = df[input_columns].apply(lambda x: too_small_or_empty(set(x), after_x_in_a_row), axis=1)
    if remove_dirty:
        return df[df['is_dirty'] == False]
    else:
        return df


def row_has_same_time(row, number_of_locs):
    '''
    Returns whether the row has all times within 10 minutes of the base location
    '''
    ten_minutes = 60 * 10 * 10**9
    for loc_number in range(2, number_of_locs + 1):
        diff = abs(
            row[0]['loc_1_measurement_datetime'].value -
            row[loc_number - 1]['loc_{}_measurement_datetime'.format(loc_number)].value
        )
        if diff > ten_minutes:
            return False
    return True


def get_loc_with_smallest_time(row, number_of_locs):
    row_names = ['loc_{}_measurement_datetime'.format(i)
                 for i in range(1, number_of_locs + 1)]
    row_times = [d[row_names[i]] for i, d in enumerate(row)]
    return min(enumerate(row_times), key=itemgetter(1))[0] + 1


def loc_has_no_more_data(row, loc_index):
    return np.isnan(row[loc_index - 1]['loc_{}_id'.format(loc_index)])


def shift_single_loc_up(df, shift_index):
    df = df[:shift_index].append(df[shift_index:].shift(-1))
    return df


def join_dataframes(dfs):
    df = None
    for d in dfs:
        if df is not None:
            df = df.join(d)
        else:
            df = d
    return df


def align_multi_location_time_series_data(dfs, number_of_locations):
    '''
    Aligns the dataframe so that each row is data for approximately the same
    point in time for all cities on that row. Returns the data frame and
    the range of every continguous series by time of rows that didn't
    need to be re-aligned.
    '''
    continuous_time_series = []
    start_index = 0
    current_index = 0
    while dfs[0].count()[0] > current_index:
        current_index += 1
        row = [df.loc[current_index] for df in dfs]
        if not row_has_same_time(row, number_of_locations):
            continuous_time_series.append((start_index, current_index))
            while not row_has_same_time(row, number_of_locations):
                loc_to_shift = get_loc_with_smallest_time(row, number_of_locations)
                dfs[loc_to_shift - 1] = shift_single_loc_up(dfs[loc_to_shift - 1], current_index)
                row = [df.loc[current_index] for df in dfs]
                #print(row)
                if loc_has_no_more_data(row, loc_to_shift):
                    return join_dataframes(dfs), continuous_time_series
            start_index = current_index
    continuous_time_series.append((start_index, current_index))
    return join_dataframes(dfs), continuous_time_series
